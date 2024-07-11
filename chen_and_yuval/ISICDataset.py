import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(128, 128), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        self.image_names = [v for v in os.listdir(image_dir) if v.endswith('.jpg')]

        if self.transform is None:
            self.transform = A.Compose([
                ToTensorV2()
            ])


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Read the image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure images are in the correct format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = np.array(image) / 255.0

        mask = cv2.resize(mask, self.img_size)
        mask = np.array(mask) / mask.max()

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        mask = (mask > 0.5).to(torch.float32).unsqueeze(0)  # Add channel dimension
        image = image.to(torch.float32)

        return image, mask

