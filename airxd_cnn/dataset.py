import os
import imageio as iio
import numpy as np
from glob import glob
from pathlib import Path

class Dataset:

    def __init__(self, n=1, shape=(2880,2880)):
        ''' n is the number of experiments. '''
        self.n = n
        self.shape = shape
        self.images = {}
        self.labels = {}

    def get_data(self, directory_names, pruned_names, image_ext='.tif', label_ext='.tif'):
        ''' get images from directories (directory_names). '''
        msg = "The number of experiments (n) doesn't match with number of directories. "
        assert self.n == len(directory_names), msg

        for i, path in enumerate(directory_names):
            ipath = sorted(glob(os.path.join(path, f'*{image_ext}')))
            lpath = sorted(glob(os.path.join(path, 'masks', f'*{label_ext}')))
            
            #Apply pruning based on pre-calculated list. Results can be re-created in example_training.ipynb step 1
            #If no pruning (i.e. pruned_list is empty), default to include everything
            if pruned_names:
                valid_input_paths = [
                    file for file in ipath
                    if any(Path(file).stem.startswith(base) for base in pruned_names)]
                valid_mask_paths = [
                    file for file in lpath
                    if any(Path(file).stem.startswith(base) for base in pruned_names)]
            else:
                valid_input_paths = ipath
                valid_mask_paths = lpath

            self.images[i] = np.zeros((len(valid_input_paths), self.shape[0], self.shape[1]))
            self.labels[i] = np.zeros((len(valid_mask_paths), self.shape[0], self.shape[1]))
            
            # get images
            for j, ip in enumerate(valid_input_paths):
                self.images[i][j] += iio.v2.volread(ip)

            # get labels
            for j, lp in enumerate(valid_mask_paths):
                print(lp)
                if label_ext == '.tif': 
                    self.labels[i][j] += iio.v2.volread(lp)
                elif label_ext == '.npy':
                    self.labels[i][j] += np.load(lp)
                else:
                    msg = "The label_ext is not implemented"
                    raise NotImplementedError(msg)

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


