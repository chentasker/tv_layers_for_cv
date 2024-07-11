import glob
import os
import sys

import joblib
import numpy as np
import torch
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from torch import nn

from chen_and_yuval.ISICDataset import ISICDataset

sys.path.append('/home/chent/Desktop/tv_layers_for_cv')
from tv_opt_layers.layers import general_tv_2d_layer
import segmentation_models_pytorch as smp

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

def calculate_metrics(gt_mask, pred_mask):
    # Flatten the masks to calculate metrics
    gt_mask_flat = gt_mask.flatten()
    pred_mask_flat = pred_mask.flatten()

    # Calculate metrics
    iou = jaccard_score(gt_mask_flat, pred_mask_flat)
    dice = f1_score(gt_mask_flat, pred_mask_flat)
    precision = precision_score(gt_mask_flat, pred_mask_flat)
    recall = recall_score(gt_mask_flat, pred_mask_flat)

    return iou, dice, precision, recall

files = glob.glob('good_logs_and_models/*trained_model.pkl')
tv_modes = [f.split('/')[-1].split('_')[0] for f in files] # ['none', 'sharp', 'smooth']
models = [joblib.load(f) for f in files]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_main_dir = os.path.join('..', 'ISIC_data')
img_size = (128, 128)
test_dataset = ISICDataset(os.path.join(image_main_dir, 'Test_Input'),
                           os.path.join(image_main_dir, 'Test_GroundTruth'), img_size=img_size)

all_iou = {tv_mode:[] for tv_mode in tv_modes}
all_dice = {tv_mode:[] for tv_mode in tv_modes}
all_precision = {tv_mode:[] for tv_mode in tv_modes}
all_recall = {tv_mode:[] for tv_mode in tv_modes}
print(len(test_dataset))
for i in range(len(test_dataset)):
    image, mask = test_dataset[i]
    image = image.reshape(1, 3, 128, 128).to(device)
    for model, tv_mode in zip(models, tv_modes):
        output = model(image).reshape(1, 128, 128) > 0
        iou, dice, precision, recall = calculate_metrics(mask.cpu().detach().numpy(), output.cpu().detach().numpy())
        all_iou[tv_mode].append(iou)
        all_dice[tv_mode].append(dice)
        all_precision[tv_mode].append(precision)
        all_recall[tv_mode].append(recall)

    print(f'i = {i}')

for tv_mode in tv_modes:
    all_iou[tv_mode] = np.mean(all_iou[tv_mode])
    all_dice[tv_mode] = np.mean(all_dice[tv_mode])
    all_precision[tv_mode] = np.mean(all_precision[tv_mode])
    all_recall[tv_mode] = np.mean(all_recall[tv_mode])

print("IOU: ", all_iou)
print("DICE: ", all_dice)
print("PERCISION: ", all_precision)
print("RECALL: ", all_recall)
