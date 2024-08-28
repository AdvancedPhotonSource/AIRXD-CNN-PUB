import os
import scipy
import numpy as np
from time import time
import importlib

import math
import random
import torch
from torch import nn, optim


import dlsia
importlib.reload(dlsia)
from dlsia.core.networks import smsnet, tunet
from dlsia.core import train_scripts
#Needed to modify source code for gpu inference.
#Some arrays were kept on cpu side which made it impossible to do
#stitching on the gpu (which is much faster than cpu)
from .qlty_modified import NCYXQuilt
import einops


import imageio as iio

class ARIXD_CNN:
    def __init__(self, model_params, training_params, quilter_params):

        #Set parameters
        self.model_params = self.update_model_parameters(model_params)
        self.training_params = training_params
        self.quilter = self.set_quilter(quilter_params)
        self.model = self.set_model(self.model_params)
        
        self.device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")

    def update_model_parameters(self, mparams):

        _mparams = {'image_shape': (2880, 2880),
                    'in_channels': 1,
                    'out_channels': 2,
                    'base_channels': 8,
                    'growth_rate': 2,
                    'depth': 4, 
                    }
        
        if mparams:
            for k, v in mparams.items():
                if k not in _mparams:
                    msg = f'The model parameter key ({k}) is invalid.'
                    raise NotImplementedError(msg)
            _mparams.update(mparams)

        return _mparams
    
    def set_model(self, mparams):
        model = tunet.TUNet(image_shape=mparams['image_shape'],
                            in_channels=mparams['in_channels'],
                            out_channels=mparams['out_channels'],
                            base_channels=mparams['base_channels'],
                            growth_rate=mparams['growth_rate'],
                            depth=mparams['depth'])
        #total_params = sum(param.numel() for param in model.parameters())
        #print(total_params)
        if self.training_params["multi_gpu"]:
            model = nn.DataParallel(model)

        return model
    
    def set_quilter(self, qparams):
        return NCYXQuilt(Y=qparams['Y'], X=qparams['X'],
                                window=qparams['window'],
                                step=qparams['step'],
                                border=qparams['border'],
                                border_weight=qparams['border_weight'])

    
    def save(self, file='./model.pt'):
        torch.save(self.model, file)

    def load(self, file):
        self.model = torch.load(file,
                                map_location=self.training_params['device'])
    
    def train(self, tloader, vloader):

        from time import time

        #Check if save_path in training params directory exists
        if not os.path.exists(self.training_params['save_path']):
            os.makedirs(self.training_params['save_path'])

        
        weights = torch.Tensor(self.training_params['weights']).to(self.training_params['device'])
        criterion = nn.CrossEntropyLoss(weight = weights, ignore_index=-1)
        minim = optim.Adam(self.model.parameters(), lr = self.training_params['lr_rate'])

        print("====================== Training ======================")
        t0 = time()
        self.model, self.res = train_scripts.train_segmentation(net =  self.model.to(self.device),
                                                                trainloader = tloader,
                                                                validationloader = vloader,
                                                                NUM_EPOCHS = self.training_params['epoch'],
                                                                criterion = criterion,
                                                                optimizer = minim,
                                                                device = self.training_params['device'],
                                                                use_amp = self.training_params['amp'],
                                                                clip_value = self.training_params['clip_value'],
                                                                savepath= self.training_params['save_path'],
                                                                show=1)

        # tloader, vloader = None, None
        t1 = time()
        t = round(t1-t0, 2)
        print("\nTotal training time: ", t, " seconds")
        print("======================================================\n")

    def predict(self, input_dset, idx):
        """
        Predicts segmemntation of image.
        Uses memory map implementation, single image at a time
        Gradually stitches predictions together to from final image
        """

        #Get shape params
        shape = input_dset[idx].shape

        #Get relevant step parameters to "re-stitch" the prediction
        nsteps_per_im = input_dset.n_steps_per_im
        x_steps_per_im = input_dset.x_steps_per_im
        y_steps_per_im = input_dset.y_steps_per_im
        window_size = input_dset.window_size

        #Pre-allocate final image 
        result = np.zeros((shape[0], shape[1]), dtype=float)




        _image = torch.Tensor(input_dset[idx]).to(self.device)


    def predict_old(self, image_file, threshold=0.1):
        """
        Predicts segmemntation of image.
        Keeping the old implementation from original code.
        Could be cleaned up for clarity but keeping as is for now.
        Note: Old implementation is super ram heavy still and does not work. Need to re-do implementation
        """

        image = iio.v2.volread(image_file)
        shape = image.shape
        _image = einops.rearrange(image, "Y X -> () () Y X")
        _image = torch.Tensor(_image).to(self.device) 
        image = self.quilter.unstitch(_image)

        with torch.no_grad():
            _result = self.model(image)
        _result = nn.Softmax(dim=1)(_result)
        _result = self.quilter.stitch(_result)

        res = []
        for i, tensor in enumerate(_result):
            res.append(tensor[i].to('cpu').numpy())

        result = np.zeros((shape[0], shape[1]), dtype=float)
        #Note from Albert: result is M x C x H x W, where M can be > 1 if you're
        #stitching more than 1 image together. We are not, so
        #we index into 0

        #It seems like anything above 10% confidence is already filtered out
        result += np.array(res[0][0]<threshold, dtype=float)
        return result
    
