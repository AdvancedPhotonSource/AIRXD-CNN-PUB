import os
import math
import imageio
import numpy as np
import glob as glob
from time import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.use('Agg')

from airxd_cnn.vanilla_model import ARIXD_CNN as cmodel
from airxd_cnn.dataset import Dataset
from sklearn.metrics import confusion_matrix as CM
from airxd_cnn.transforms import create_filtered_masks

import torch

#Seeding
seed = 42
np.random.seed(seed) #Seeding for dataloader manipulation
torch.manual_seed(0) #Seeding model for training

#Get the pruned list for training
with open('data/pruned_list.pkl', 'rb') as file:
    pruned_im_list = pickle.load(file)

#Create filtered masks
base_dir = ['data']
sub_dirs = ['battery1', 'battery2', 'battery3', 'battery4', 'battery5', 'Nickel']
create_filtered_masks(base_dir, sub_dirs)

#Now create dataset for training
directories = [base_dir[0] + '/' + sub_dir for sub_dir in sub_dirs]
dataset = Dataset(n=len(directories))

dataset.get_data(directories, pruned_im_list,label_ext='.tif')

# Quilter params
N = 256
M = N // 2
B = M // 4
quilter_params = {'Y': 2880, 'X': 2880,
                  'window': (N, N),
                  'step': (M, M),
                  'border': (B, B),
                  'border_weight': 0}

# TUNet params
model_params = {'image_shape': (2880, 2880),
                'in_channels': 1,
                'out_channels': 2,
                'base_channels': 8,
                'growth_rate': 2,
                'depth': 3}

# Training params
epoch = 30
batch_size = 50
lr_rate = 1e-2

# Training
model = cmodel(quilter_params, model_params, device='cuda:0')
# We've already pruned the data here. So we include all valid data and associated thresholded masks and train.
model.train(dataset, include_data='all',
            epoch=epoch,
            batch_size=batch_size,
            lr_rate=lr_rate)
model.save('./test_model.pt')

