import torchvision.transforms.functional as TF
import numpy as np
import random
import os
import imageio as iio

#Import Dataset from torch
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF
import PIL




class powder_dset(Dataset):
    """
    Reads input powder diffraction patterns and corresponding masks
    Combines them into tensor pair for dataloader

    Applies subcropping, rotation and flipping to both mask/image simultaneously.
    Subcropping will be done via indexing method similar to qlty. Indexing window will be calculated from
    a simple index number, and based on image cropping parameters set in init. Going to be very similar
    to what is done in qlty.
    """

    def __init__(self,
                 input_paths,
                 target_paths,
                 **kwargs
                 ):
        """
        Args:
            root_path (string): Path to root directory of dataset
            input_path (string): Path to input images
            target_path (string): Path to target images
            train_transform (bool, optional): Optional transform to be applied
        """
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform = kwargs["transforms"]
        self.dir_len = len(self.input_paths)

        #Pre-calculate image windowing based on qlty
        self.x_steps_per_im = self.calculate_steps(kwargs["im_size"][-2], kwargs["window_size"][-2])
        self.y_steps_per_im = self.calculate_steps(kwargs["im_size"][-1], kwargs["window_size"][-1])
        self.nsteps_per_im = self.x_steps_per_im * self.y_steps_per_im
        #Other useful parameters
        self.im_size = kwargs["im_size"]
        self.window_size = kwargs["window_size"]


    def calculate_steps(self, im_size, window_size):
        step_size = window_size // 2
        full_steps = (im_size - window_size) // step_size

        # if im_size > full_steps * step_size + window_size:
        #     return full_steps + 2
        # else:
        return full_steps + 1
        
    def linear_index_to_window(self, index):
        #Select specific image from input_paths
        #Index is 1-indexed (not zero indexed)
        #E.g. each image will have 484 windows, so index of 500 would pick
        #the second image on the input_path

        image_id = (index-1) // self.nsteps_per_im + 1

        #Calculate x and y index of window
        #image_index is a linear index
        image_index = (index-1) % self.nsteps_per_im
        y_index = image_index // self.x_steps_per_im
        x_index = image_index % self.x_steps_per_im

        #Calculate x and y start, endpoints
        x_start = x_index * self.window_size[0]//2
        y_start = y_index * self.window_size[1]//2

        return image_id, x_start, y_start

    #Return length of dataset    
    def __len__(self):
        return self.dir_len * self.x_steps_per_im * self.y_steps_per_im
    
    #Get item using indexing and subcropping
    def __getitem__(self, index):
        #Get image id, x and y start points
        image_id, x_start, y_start = self.linear_index_to_window(index)

        #Load images using iio.v2.volread
        #Input
        #Time each step
        x = iio.v2.volread(self.input_paths[image_id])
        # x = PIL.Image.open("../" + self.input_paths[image_id])
        x = ToTensor()(x)
        x = TF.crop(x, x_start, y_start, self.window_size[0], self.window_size[1])
        
        #Target
        t = iio.v2.volread(self.target_paths[image_id])
        t = ToTensor()(t)
        t = TF.crop(t, x_start, y_start, self.window_size[0], self.window_size[1])

        #Transform with whatever is in the pipeline
        if self.transform is not None:
            x, t = self.transform(x,t)
       

        return x, t

    


def parse_imctrl(filename):
    controls = {'size': [2880, 2880], 'pixelSize': [150.0, 150.0]}
    keys = ['IOtth', 'PolaVal', 'azmthOff', 'rotation', 'distance', 'center', 'tilt', 'DetDepth']
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ln = line.split(':')
            if ln[0] in keys:
                if ln[1][0] == '[':
                    temp = []
                    temp_list = ln[1].split(',')
                    temp.append(float(temp_list[0][1:]))
                    try:
                        temp.append(float(temp_list[1][:-2]))
                    except:
                        temp.append(False)
                    controls[ln[0]] = temp
                else:
                    controls[ln[0]] = float(ln[1])

    return controls 