from typing import List, Callable, Tuple
import numpy as np

#For normalization
import skimage
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.filters import rank
from skimage.morphology import disk


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

    def __call__(self, inp: np.ndarray, tar: dict):
        if self.input: inp = self.function(inp)
        if self.target: tar = self.function(tar)
        return inp, tar


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target
    
def rotate_random(image: np.ndarray) -> np.ndarray:
    '''
    Randomly rotate image by 0, 90, 180, or 270 degrees
    '''
    angle = np.random.choice([0, 1, 2, 3])
    return np.rot90(image, k=angle)
def powder_normalize(image: np.darray) -> np.darray:
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
