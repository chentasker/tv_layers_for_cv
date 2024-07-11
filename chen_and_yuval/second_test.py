import joblib
import segmentation_models_pytorch as smp
import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2


sys.path.append('/home/chent/Desktop/tv_layers_for_cv')
from tv_opt_layers.layers import general_tv_2d_layer
import matplotlib.pyplot as plt
from ISICDataset import ISICDataset

"""
prefix = input("Prefix For Files:")
log_name = prefix + "_training_log.txt"
pkl_name = prefix + "_trained_model.pkl"
"""

class TVNet(nn.Module):
    def __init__(self, device, tv_mode):
        super(TVNet, self).__init__()
        if tv_mode is None or tv_mode == 'none':
            self.tv_layer = nn.Identity()
        else:
            self.tv_layer = general_tv_2d_layer.GeneralTV2DLayer(num_channels=3, filt_mode=tv_mode).to(device)
        self.seg_model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        ).to(device)

    def forward(self, x):
        if x.shape[1] != 3 or x.shape[2] != 128 or x.shape[3] != 128:
            print(x.shape)
        return self.seg_model(self.tv_layer(x))


# Define the size you want to resize your images
img_size = (128, 128)

# Data augmentation
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30)
])

# Define the augmentations
data_transforms = A.Compose([
    A.RandomCrop(width=img_size[0], height=img_size[1]),
    A.Resize(width=img_size[0], height=img_size[1]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
    A.GaussNoise(p=0.5),
    ToTensorV2()
])

# Create DataLoader with multiple workers
print('Creating DataLoader...', end=' ')
image_main_dir = os.path.join('..', 'ISIC_data')
batch_size = 32
num_workers = 4  # Adjust based on your CPU cores
train_dataset = ISICDataset(os.path.join(image_main_dir, 'Training_Input'),
                            os.path.join(image_main_dir, 'Training_GroundTruth'), img_size=img_size,
                            transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataset = ISICDataset(os.path.join(image_main_dir, 'Validation_Input'),
                          os.path.join(image_main_dir, 'Validation_GroundTruth'), img_size=img_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_dataset = ISICDataset(os.path.join(image_main_dir, 'Test_Input'),
                           os.path.join(image_main_dir, 'Test_GroundTruth'), img_size=img_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('Done.')

# Initialize the model, criterion, and optimizer
print('Initializing model, criterion and optimizer...', end=' ')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()
print('Done.')
print(f'Device: {device}')

# ['none', 'sharp', 'smooth']
for tv_mode in ['none']:
    log_name = tv_mode + "_training_log.txt"
    pkl_name = tv_mode + "_trained_model.pkl"
    checkpoint_name = tv_mode + "_checkpoint.pth"
    if False:
        model = joblib.load(pkl_name)
    else:
        num_epochs = 50
        model = TVNet(device=device, tv_mode=tv_mode)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        start_epoch = 0
        try:
            checkpoint = torch.load(checkpoint_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f'Resuming training from epoch {start_epoch+1}')
        except FileNotFoundError:
            print('No checkpoint found, starting from scratch')

        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            train_str = f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}"
            print(train_str)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader.dataset)

            val_str = f"Validation Loss: {val_loss:.4f}"
            print(val_str)

            with open(log_name, 'a') as f:
                f.write(train_str)
                f.write(val_str)
                f.write('\n')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_name)

        # save model parameters
        joblib.dump(model, pkl_name)

i = 0
while True:
    fig, axes = plt.subplots(5, 6)
    indices = list(range(i * 10, (i + 1) * 10))
    all_images = []
    for ind in indices:
        image, mask = test_dataset[ind]
        image = image.reshape(1, 3, 128, 128).to(device)
        output = model(image).reshape(1, 128, 128) > 0
        all_images.extend([np.transpose(image.reshape(3, 128, 128).cpu().detach().numpy(), (1, 2, 0)),
                           np.transpose(mask.cpu().detach().numpy(), (1, 2, 0)),
                           np.transpose(output.cpu().detach().numpy(), (1, 2, 0))])
    for ind, ax in enumerate(axes.flat):
        ax.imshow(all_images[ind])
        ax.axis('off')
    plt.show(block=True)
    i = i + 1
