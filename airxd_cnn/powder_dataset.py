import torchvision.transforms.functional as TF
import numpy as np
import random
import os

#Import Dataset from torch
from torch.utils.data import Dataset


class powder_dset(Dataset):
    """
    Reads input powder diffraction patterns and corresponding masks
    Combines them into tensor pair for dataloader

    Applies subcropping, rotation and flipping to both mask/image simultaneously.
    Attempts to combat data imbalance by recropping image if mask has too few minority samples
    """

    def __init__(self,
                 input_paths,
                 target_paths,
                 train_transform=None):
        """
        Args:
            root_path (string): Path to root directory of dataset
            input_path (string): Path to input images
            target_path (string): Path to target images
            train_transform (bool, optional): Optional transform to be applied
        """
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.train_transform = train_transform

    def transform_training(self):
        """"""
    #Return length of dataset    
    def __len__(self):
        return len(self.input_paths)
    
    #
    

