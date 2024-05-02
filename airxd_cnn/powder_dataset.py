import torchvision.transforms.functional as TF
import numpy as np
import random
import os
import imageio as iio

#Import Dataset from torch
import torch
from torch.utils.data import Dataset




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
                 transform=None,
                 window_size = (256,256),
                 im_size = (2880,2880)):
        """
        Args:
            root_path (string): Path to root directory of dataset
            input_path (string): Path to input images
            target_path (string): Path to target images
            train_transform (bool, optional): Optional transform to be applied
        """
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform = transform
        self.len = len(self.input_paths)

        #Pre-calculate image windowing based on qlty
        self.x_steps_per_im = self.calculate_steps(im_size[-2], window_size[-2])
        self.y_steps_per_im = self.calculate_steps(im_size[-1], window_size[-1])

        #Other useful parameters
        self.im_size = im_size
        self.window_size = window_size


    def calculate_steps(self, im_size, window_size):
        step_size = window_size // 2
        full_steps = (im_size - window_size) // step_size

        if im_size > full_steps * step_size + window_size:
            return full_steps + 2
        else:
            return full_steps + 1
        
    def linear_index_to_window(self, index):
        #Select specific image from input_paths
        #E.g. each image will have 484 windows, so index of 500 would pick
        #the second image on the input_path
        image_id = index // (self.x_steps_per_im * self.y_steps_per_im)

        #Calculate x and y index of window
        image_index = index % (self.x_steps_per_im * self.y_steps_per_im)
        y_index = image_index // self.x_steps_per_im
        x_index = image_index % self.x_steps_per_im

        #Calculate x and y start, endpoints
        x_start = x_index * self.window_size[0]
        y_start = y_index * self.window_size[1]

        return image_id, x_start, y_start

    #Return length of dataset    
    def __len__(self):
        return self.len
    
    #Get item using indexing and subcropping
    def __getitem__(self, index):
        #Get image id, x and y start points
        image_id, x_start, y_start = self.linear_index_to_window(index)

        #Load images using iio.v2.volread
        x = iio.v2.volread(self.input_paths[image_id])
        x = x[x_start:x_start+self.window_size[0], y_start:y_start+self.window_size[1]]

        y = iio.v2.volread(self.target_paths[image_id])
        y = y[x_start:x_start+self.window_size[0], y_start:y_start+self.window_size[1]]

        #Transform with whatever is in the pipeline
        if self.transform is not None:
            x, y = self.transform(x, y)

        #Typecast to torch array
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y

    


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