from typing import List, Callable, Tuple
import numpy as np
from glob import glob
import os
import imageio as iio

#For normalization
import skimage
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.filters import rank
from skimage.morphology import disk
from scipy.ndimage import label
from skimage.measure import regionprops

import torch
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF



#Define transforms using flexible pipeline:

#Taken from: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55

class Repr:
    """Evaluable string representation of an object"""

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'


class FunctionWrapperSingle(Repr):
    """A function wrapper that returns a partial for input only."""

    def __init__(self, function: Callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray): return self.function(inp)


class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(self, function: Callable, input: bool = True, target: bool = False, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        if self.input and self.target:
            inp, tar = self.function(inp, tar)
        elif self.input and not self.target:
            inp = self.function(inp)
        return inp, tar


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp: np.ndarray, target: np.ndarray):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target
    
#Trying out custom transform with nn.module
#Source: https://pytorch.org/vision/stable/auto_examples/transforms/plot_custom_transforms.html#sphx-glr-auto-examples-transforms-plot-custom-transforms-py

#Rotation
class RandomRotation(torch.nn.Module):
    "Rotate image and target by 0, 90, 180, or 270 degrees"
    def forward(self, image: torch.Tensor, target: torch.Tensor):
        angle = np.random.choice([0, 1, 2, 3])
        #Use TF rotate to customize rotation behavior
        return TF.rotate(image, angle*90), TF.rotate(target, angle*90)
    
#Random horizontal/vertical flip
class RandomFlip(torch.nn.Module):
    "Randomly flip image horizontally or vertically"
    def forward(self, image: torch.Tensor, target: torch.Tensor):
        flip = np.random.choice([0, 1, 2])
        if flip == 0:
            return image, target
        elif flip == 1:
            return TF.horizontal_flip(image), TF.horizontal_flip(target)
        else:
            return TF.vertical_flip(image), TF.vertical_flip(target)





#Old functions    
def rotate_random(image: np.ndarray, tar: np.ndarray):
    '''
    Randomly rotate image and target by 0, 90, 180, or 270 degrees
    '''
    angle = np.random.choice([0, 1, 2, 3])
    return np.rot90(image, k=angle), np.rot90(tar, k=angle)

def flip_random(image: np.ndarray, tar: np.ndarray):
    '''
    Randomly flip image horizontally or vertically
    '''

    flip = np.random.choice([0, 1, 2])
    if flip == 0:
        return image, tar
    elif flip == 1:
        return np.fliplr(image), np.fliplr(tar)
    else:
        return np.flipud(image), np.flipud(tar)

#Don't touch
def powder_normalize(image: np.ndarray) -> np.ndarray:
    '''
    Pre-processing powder normalization; not included in pipeline
    '''
    footprint = disk(32)
    img = np.log(np.abs(image) - np.min(image) + 1e-7)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    img = skimage.util.img_as_ubyte(img) 

    #Most important step. Spreads out brightest pixels to enhance low contrast
    img_eq = rank.equalize(img, footprint)
    img_eq = img_eq.astype(float)/256.0
    return img_eq

def filter_connected_components_by_size(mask, min_size, max_size=None):
    """
    Filters connected components in the mask based on size.
    
    Parameters:
        mask (ndarray): Binary mask where spots are marked as 1, and the background is 0.
        min_size (int): Minimum number of pixels for a connected component to be kept.
        max_size (int, optional): Maximum number of pixels for a connected component to be kept. If None, no upper limit.
        
    Returns:
        filtered_mask (ndarray): Mask with only the connected components that meet the size criteria.
    """
    # Label each connected component in the binary mask
    labeled_mask, _ = label(mask)
    
    # Initialize an empty mask to hold the filtered components
    filtered_mask = np.zeros_like(mask)
    
    # Get properties of labeled regions
    regions = regionprops(labeled_mask)
    
    for region in regions:
        # Check the size of each region
        region_size = region.area
        if region_size >= min_size and (max_size is None or region_size <= max_size):
            # Add this region to the filtered mask
            filtered_mask[labeled_mask == region.label] = 1
    
    return filtered_mask

def create_filtered_masks(base_dir, sub_dirs):

    for materials in sub_dirs:
        image_dir = base_dir + '/' + materials

        #Get sub image list
        mask_list = glob(image_dir + '/masks/*.tif')
        thresholded_mask_dir = image_dir + '/thresholded_mask'
        
        #Create directory for masked directories
        if not os.path.exists(thresholded_mask_dir):
            os.mkdir(thresholded_mask_dir)
        
            print(f'Masking images for material {materials}')
            #For each material take each mask in the mask list and run the island filter on it
            
            for mask_path in mask_list:
                mask = iio.v2.volread(mask_path)
                thresholded_mask = filter_connected_components_by_size(mask, 5)

                thresholded_mask_path = os.path.join(thresholded_mask_dir, os.path.split(mask_path)[-1])

                #Write new image
                iio.imwrite(thresholded_mask_path, thresholded_mask)