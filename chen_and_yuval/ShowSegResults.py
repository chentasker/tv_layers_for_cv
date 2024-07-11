import glob
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('/home/chent/Desktop/tv_layers_for_cv')
from tv_opt_layers.layers import general_tv_2d_layer
import segmentation_models_pytorch as smp

from chen_and_yuval.ISICDataset import ISICDataset

from second_test import TVNet

files = glob.glob('good_logs_and_models/*trained_model.pkl')
tv_modes = [f.split('/')[-1].split('_')[0] for f in files] # ['none', 'sharp', 'smooth']
models = [joblib.load(f) for f in files]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_main_dir = os.path.join('..', 'ISIC_data')
img_size = (128, 128)
test_dataset = ISICDataset(os.path.join(image_main_dir, 'Test_Input'),
                           os.path.join(image_main_dir, 'Test_GroundTruth'), img_size=img_size)

start_i = 0
while True:
    #plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(5, len(files)+2)

    # Column titles
    column_titles = ['Input Image'] + [f'TV mode: {mode}' for mode in tv_modes] + ['Ground Truth']
    for ax, col in zip(axes[0], column_titles):
        ax.set_title(col, fontsize=12)

    all_all_images = []
    for i in range(start_i, start_i+5):
        all_images = []
        image, mask = test_dataset[i]
        image = image.reshape(1, 3, 128, 128).to(device)
        for model, tv_mode in zip(models, tv_modes):
            output = model(image).reshape(1, 128, 128) > 0
            all_images.extend([np.transpose(output.cpu().detach().numpy(), (1, 2, 0))])
        all_images = [np.transpose(image.reshape(3, 128, 128).cpu().detach().numpy(), (1, 2, 0))] + all_images + [np.transpose(mask.cpu().detach().numpy(), (1, 2, 0))]
        all_all_images = all_all_images + all_images

    for ind, ax in enumerate(axes.flat):
        ax.imshow(all_all_images[ind])
        ax.axis('off')
    plt.show(block=True)
    start_i = start_i + 5