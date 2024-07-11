import os

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from tv_opt_layers.layers import general_tv_2d_layer
from ISICDataset import ISICDataset

prefix = input("Prefix For Files:")
log_name = prefix + "_training_log.txt"
pkl_name = prefix + "_trained_model.pkl"

class UNet(nn.Module):
    def __init__(self, device, tv_mode=None):
        super(UNet, self).__init__()

        if tv_mode is None:
            self.tv_layer = nn.Identity()
        else:
            self.tv_layer = general_tv_2d_layer.GeneralTV2DLayer(num_channels=3, filt_mode=tv_mode).to(device)

        self.initial_conv = self.conv_block(3, 64)

        self.encoder1 = self.conv_block(64, 128)
        self.encoder2 = self.conv_block(128, 256)
        self.encoder3 = self.conv_block(256, 512)
        self.encoder4 = self.conv_block(512, 1024)

        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)

        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.tv_layer(x)
        # Encoder path
        x1 = self.initial_conv(x)

        x2 = self.encoder1(F.max_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.encoder2(F.max_pool2d(x2, kernel_size=2, stride=2))
        x4 = self.encoder3(F.max_pool2d(x3, kernel_size=2, stride=2))
        x5 = self.encoder4(F.max_pool2d(x4, kernel_size=2, stride=2))

        # Decoder path
        x = self.upconv4(x5)
        x = torch.cat((x, x4), dim=1)
        x = self.decoder4(x)

        x = self.upconv3(x)
        x = torch.cat((x, x3), dim=1)
        x = self.decoder3(x)

        x = self.upconv2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.decoder2(x)

        x = self.upconv1(x)
        x = torch.cat((x, x1), dim=1)
        x = self.decoder1(x)

        x = self.final_conv(x)
        return torch.sigmoid(x)


# Define the size you want to resize your images
img_size = (128, 128)


# Data augmentation
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30)
])

# Create DataLoader with multiple workers
print('Creating DataLoader...', end=' ')
image_main_dir = os.path.join('..', 'ISIC_data')
batch_size = 32
num_workers = 4  # Adjust based on your CPU cores
train_dataset = ISICDataset(os.path.join(image_main_dir, 'Training_Input'),
                            os.path.join(image_main_dir, 'Training_GroundTruth'), img_size=img_size, transform=data_transforms)
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
model = UNet(device).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
print('Done.')
print(f'Device: {device}')

if False:
    model = joblib.load(pkl_name)
else:
    num_epochs = 30

    for epoch in range(num_epochs):
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

    # save model parameters
    joblib.dump(model, pkl_name)


i = 0
while True:
    fig, axes = plt.subplots(5, 6)
    indices = list(range(i*10, (i+1)*10))
    all_images = []
    for ind in indices:
        image, mask = test_dataset[ind]
        image = image.reshape(1,3,128,128).to(device)
        output = model(image).reshape(1,128,128) > 0
        all_images.extend([np.transpose(image.reshape(3,128,128).cpu().detach().numpy(), (1,2,0)),
                           np.transpose(mask.cpu().detach().numpy(), (1, 2, 0)),
                           np.transpose(output.cpu().detach().numpy(), (1, 2, 0))])
    for ind, ax in enumerate(axes.flat):
        ax.imshow(all_images[ind])
        ax.axis('off')
    plt.show(block=True)
    i = i+1

