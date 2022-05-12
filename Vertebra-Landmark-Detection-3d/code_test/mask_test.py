import joblib
import json
import torch
import numpy as np
import sys
import os

from torch._C import dtype
o_path = os.getcwd()
sys.path.append(o_path)
from draw.draw import draw_points_test, draw_by_matplotlib
#import draw_distribute_points
from models import spinal_net
import SimpleITK as sitk
from utils import decoder
from preprocess import transform
import torch.nn as nn
from dataset.dataset import BaseDataset
import loss
from matplotlib import pyplot as plt

if __name__ == '__main__':
    mask_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/Verse2020_processed/sub-verse504_dir-iso_ct.nii.gz'
    mask_path = r'/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/dataset-verse20training/dataset-01training/rawdata/sub-verse504/sub-verse504_dir-iso_ct.nii.gz'
    mask_path = r'/home/gpu/Spinal-Nailing/ZN-CT-nii/data/k_fold_smooth/64.nii.gz'
    itk_img = sitk.ReadImage(mask_path)
    print(itk_img.GetSize())
    image_array = sitk.GetArrayFromImage(itk_img)
    image_array_section = image_array[229]
    plt.imshow(image_array_section, cmap='gray')
    plt.scatter(266, 266, s=80, c='red')
    plt.show()

    plt.imshow(image_array[:,266,:], cmap='gray')
    plt.show()
    plt.imshow(image_array[:, :, 266], cmap='gray')
    plt.show()
    print('ok')
    #print(image_array[192,352,257],image_array[234,352,257],image_array[276,352,257],image_array[315,352,257],image_array[350,352,257],image_array[387,352,257])