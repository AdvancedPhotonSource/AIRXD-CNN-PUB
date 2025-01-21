from sklearn.utils import gen_batches
import torchvision.transforms.functional as TF
import numpy as np
import random
import os
import imageio as iio
import shutil
from functools import partial

#Torch
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF

#Image manipulation
import PIL
from mmap_ninja.ragged import RaggedMmap
from qlty import qlty2D
import einops
import pickle



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
                 qlty_params,
                 **kwargs
                 ):
        """
        Args:
            root_path (string): Path to root directory of dataset
            input_path (string): Path to input images
            target_path (string): Path to target images
            train_transform (bool, optional): Optional transform to be applied
        """
        #Paths
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.dir_len = len(self.input_paths)

        #2D unstitching
        self.quilter = self.set_quilter(qlty_params)
        self.n_patches = self.get_n_patches(qlty_params)
        self.device = kwargs["device"]

        #Transform pipeline
        self.crop = qlty_params["crop"]
        self.transform = kwargs["transforms"]

        #For minority class training
        self.good_indices = []
        self.bad_indices = []
        self.minority_threshold = kwargs["minority_threshold"]

        #Create memory maps
        self.input_map_path = kwargs["input_map_path"]
        self.target_map_path = kwargs["target_map_path"]
        
        if kwargs['create_memmap']:
            self.memory_map(input_paths, target_paths)
            #Use pickle to save self.good_incies and self.bad_indices
            with open("data/indices.pkl", 'wb') as f:
                pickle.dump([self.good_indices, self.bad_indices], f)
        else:
            #Load in indices
            with open("data/indices.pkl", 'rb') as f:
                self.good_indices, self.bad_indices = pickle.load(f)

        self.inputs = RaggedMmap(self.input_map_path)
        self.targets = RaggedMmap(self.target_map_path)

    def memory_map(self,input_paths, target_paths):
        """
        Create a memory map of the input and target paths using memmap_ninja
        """
        #Print status message
        print("Creating memory map of input and target paths...")

        if os.path.exists(self.input_map_path):
            shutil.rmtree(self.input_map_path)
        if os.path.exists(self.target_map_path):
            shutil.rmtree(self.target_map_path)

        #Clumsy good indice implementation. Don't want to make a new function for this.
        #Input
        self.add_to_list = False
        RaggedMmap.from_generator(
            out_dir=self.input_map_path,
            sample_generator=map(self.generate_patch, input_paths),
            batch_size=4,
            verbose=True
        )

        #Target
        self.i = 0
        self.add_to_list = True
        RaggedMmap.from_generator(
            out_dir = self.target_map_path,
            sample_generator=map(self.generate_patch, target_paths),
            batch_size=4,
            verbose=True
        )
        print('Done!')

    def generate_patch(self, image_path):
        """
        Create numpy array with all the patches from a single image to store to memory.
        We are pre-computing the patches and storing them in a memory map to speed up data loading.

        We're using qlty to do the image patch unstitching
        """

        #Load in image with volread
        image = iio.v2.volread(image_path)
        shape = image.shape
        #Crop
        image = image[self.crop:shape[0]-self.crop, self.crop:shape[1]-self.crop]
        #Reshape
        _image = einops.rearrange(image, "Y X -> () () Y X")
        #Torch tensor for qlty
        _image = torch.Tensor(_image)
        #Unstitch image to patches
        patch = self.quilter.unstitch(_image)
        patch = torch.squeeze(patch)
        #Convert to numpy
        np_patch = patch.numpy()

        #Find indices where np_patch images have greater than threshold
        if self.add_to_list:
            #Sum up the number of minority labels
            np_patch_sum = np_patch.sum(axis=(1,2))
            #Good indices are ones where there are more than the minority threshold in the patch
            good_idx = np.where((np_patch_sum > self.minority_threshold))
            bad_idx = np.where(np_patch_sum <= self.minority_threshold)

            #Add indices to final index list (we can use this to control minority/majority)
            good_idx_list = (self.n_patches * self.i + good_idx[0]).tolist()
            bad_idx_list = (self.n_patches * self.i + bad_idx[0]).tolist()

            self.good_indices.extend(good_idx_list)
            self.bad_indices.extend(bad_idx_list)
            
            self.i += 1
        

        return np_patch
       
    def get_n_patches(self, qlty_params):
        """
        Computes the number of chunks along Z, Y, and X dimensions, ensuring the last chunk
        is included by adjusting the starting points.
        """
        def compute_steps(dimension_size, window_size, step_size):
            # Calculate the number of full steps
            full_steps = (dimension_size - window_size) // step_size
            # Check if there is enough space left for the last chunk
            if dimension_size > full_steps * step_size + window_size:
                return full_steps + 2
            else:
                return full_steps + 1
            
        Y_times = compute_steps(qlty_params['Y'], qlty_params['window'][-2], qlty_params['step'][-2])
        X_times = compute_steps(qlty_params['X'], qlty_params['window'][-1], qlty_params['step'][-1])

        return Y_times * X_times        
        
    #Create qlty2D
    def set_quilter(self, qparams):
        return qlty2D.NCYXQuilt(Y=qparams['Y'], X=qparams['X'],
                            window=qparams['window'],
                            step=qparams['step'],
                            border=qparams['border'],
                            border_weight=qparams['border_weight'])
        
    def linear_index_to_image_index(self, index):
        #Select specific image from input_paths
        image_id = index // self.n_patches

        #Caculate image index we're retrieving from
        image_index = index % self.n_patches

        return image_id, image_index

    #Return length of dataset    
    def __len__(self):
        return self.dir_len * self.n_patches
    
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