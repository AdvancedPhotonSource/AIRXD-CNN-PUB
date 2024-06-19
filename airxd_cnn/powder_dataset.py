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
from mmap_ninja.ragged import RaggedMmap



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

        #Create memory map
        self.input_map_path = kwargs["input_map_path"]
        self.target_map_path = kwargs["target_map_path"]
        
        if not os.path.exists( self.input_map_path) and not os.path.exists(self.target_map_path):
            self.memory_map(input_paths, target_paths)

        self.inputs = RaggedMmap(self.input_map_path)
        self.targets = RaggedMmap(self.target_map_path)


    def memory_map(self,input_paths, target_paths):
        """
        Create a memory map of the input and target paths using memmap_ninja
        """
        #Print status message
        print("Creating memory map of input and target paths...")

        #Input
        RaggedMmap.from_generator(
            out_dir='data/input_mmap',
            sample_generator=map(self.generate_patch, input_paths),
            batch_size=4,
            verbose=True
        )

        #Target
        RaggedMmap.from_generator(
            out_dir = 'data/target_mmap',
            sample_generator=map(self.generate_patch, target_paths),
            batch_size=4,
            verbose=True
        )
        print('Done!')

    def generate_patch(self, image_path):
        """
        Create numpy array with all the patches from a single image to store to memory.
        We are pre-computing the patches and storing them in a memory map to speed up data loading.
        """

        #Load in image with volread
        image = iio.v2.volread(image_path)
        #Create empty array
        patches = np.zeros((self.nsteps_per_im, self.window_size[0], self.window_size[1]))
        patches = np.float32(patches)
        #Loop through each patch
        for i in range(self.nsteps_per_im):
            #Get x and y start points
            x_start = (i % self.x_steps_per_im) * self.window_size[0]//2
            y_start = (i // self.x_steps_per_im) * self.window_size[1]//2
            #Crop image
            patches[i] = image[x_start:x_start+self.window_size[0], y_start:y_start+self.window_size[1]]

        return patches



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

    def linear_index_to_image_index(self, index):
        #Select specific image from input_paths
        image_id = index // self.nsteps_per_im

        #Caculate image index we're retrieving from
        image_index = index % self.nsteps_per_im

        return image_id, image_index

    #Return length of dataset    
    def __len__(self):
        return self.dir_len * self.x_steps_per_im * self.y_steps_per_im
    
    #Get item using indexing and subcropping
    def __getitem__(self, index):
        #Get image id, x and y start points
        image_id, image_index = self.linear_index_to_image_index(index)

        #Try loading in this image. If it doesn't work, spit out image_id
        try:
            x = self.inputs[image_id][image_index]
        except:
            print("Error loading image with id: {}".format(image_id))
        x = ToTensor()(x)

        #Target
        t = self.targets[image_id][image_index]
        t = ToTensor()(t)

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