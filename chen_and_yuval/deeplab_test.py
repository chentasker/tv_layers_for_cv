import os
import torch

import sys
sys.path.append('/home/chent/Desktop/tv_layers_for_cv/pytorch-deeplab-xception')
from modeling.deeplab import DeepLab
"""
# Import the model from the DeepLabv3+ repository
deeplab = DeepLab(num_classes=2, backbone='resnet', output_stride=16)
deeplab.eval()

# Example: Load an image and preprocess it
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

input_image = Image.open(os.path.join('ISIC_data','Test_Input','ISIC_0012236.jpg'))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

plt.imshow(input_tensor.permute(1, 2, 0).detach().numpy())
plt.show()

# Perform inference
with torch.no_grad():
    output = deeplab(input_batch)[0]
output_predictions = output.argmax(0)

# Display the results
plt.imshow(output_predictions.cpu().numpy())
plt.show()
"""

import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add the full path to the pytorch-deeplab-xception directory
sys.path.append('/home/chent/Desktop/tv_layers_for_cv/pytorch-deeplab-xception')

# Import the DeepLab model from the repo
from modeling.deeplab import DeepLab

# Initialize the model, criterion, optimizer, and scheduler
model = DeepLab(num_classes=2, backbone='resnet', output_stride=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Load your dataset
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the size you want to resize your images
img_size = (128, 128)

image_main_dir = os.path.join('ISIC_data')
from dataloaders.datasets import ISIC
train_dataset = ISIC.ISICSegmentation(os.path.join(image_main_dir, 'Training_Input'),
                            os.path.join(image_main_dir, 'Training_GroundTruth'), img_size=img_size, transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks['segmentation'].to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    scheduler.step()

    # Save the model checkpoint if needed
    torch.save(model.state_dict(), 'deeplab_model.pth')

