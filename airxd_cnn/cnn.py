import os
import scipy
import numpy as np
from time import time

import math
import random
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import imageio

import dlsia
from dlsia.core.networks import smsnet, tunet
from dlsia.core import train_scripts
from qlty import qlty2D, cleanup
import einops

import skimage
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.filters import rank
from skimage.morphology import disk

#torch.manual_seed(0)
#np.random.seed(0)
#torch.set_default_dtype(torch.float16)

class ARIXD_CNN:

    def __init__(self, quilter_params=None, model_params=None, device='cuda', amp=False, clip_value=None):
        self.device = device
        self.quilter_params = self.update_quilter_parameters(quilter_params)
        self.model_params = self.update_model_parameters(model_params)

        self.quilter = self.set_quilter(self.quilter_params)
        self.model = self.set_model(self.model_params)
        self.amp = amp
        self.clip_value = clip_value


    def set_quilter(self, qparams):
        return qlty2D.NCYXQuilt(Y=qparams['Y'], X=qparams['X'],
                                window=qparams['window'],
                                step=qparams['step'],
                                border=qparams['border'],
                                border_weight=qparams['border_weight'])

    def update_quilter_parameters(self, qparams):
        _qparams = {'Y': 2880, 'X': 2880,
                    'window': (128, 128),
                    'step': (64, 64),
                    'border': (16, 16),
                    'border_weight': 0}
        
        if qparams:
            for k, v in qparams.items():
                if k not in _qparams:
                    msg = f'The quilter parameter key ({k}) is invalid.'
                    raise NotImplementedError(msg)
            _qparams.update(qparams)
        return _qparams

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
        return model


    def train(self, dataset, include_data='random', training_images=5, 
              epoch=15,
              batch_size=64, 
              shuffle=True, 
              drop_last=False,
              lr_rate=1e-2,
              weights=[1.0, 10.0]):
              #criterion='CrossEntropyLoss', # need to put these more
              #optim='Adam'):                # coherently.
        from time import time       

        include = {}
        if include_data == 'random':
            print("Data included in training: ")
            for i in range(dataset.n):
                include[i] = random.sample(range(0, len(dataset.images[i])), training_images)
                print(i, ": ", include[i])
        else:
            include = include_data
            for k, v in include.items():
                if k in dataset.images:
                    print(k, ": ", v)
        
        self.shape = (dataset.shape[0], dataset.shape[1])
        
        weights = torch.Tensor(weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
        minim = optim.Adam(self.model.parameters(), lr=lr_rate)
        tloader, vloader = self.parse_dataset(dataset, include, batch_size, shuffle, drop_last)

        print("====================== Training ======================")
        t0 = time()
        self.model, self.res = train_scripts.train_segmentation(net=self.model.to(self.device),
                                                                trainloader=tloader,
                                                                validationloader=vloader,
                                                                NUM_EPOCHS=epoch,
                                                                criterion=criterion,
                                                                optimizer=minim,
                                                                device=self.device,
                                                                use_amp=self.amp,
                                                                clip_value=self.clip_value,
                                                                show=1)

        tloader, vloader = None, None
        t1 = time()
        t = round(t1-t0, 2)
        print("\nTotal training time: ", t, " seconds")
        print("======================================================\n")

    def predict(self, image):
        #image = imageio.volread(image_file)
        shape = image.shape
        _image = einops.rearrange(self.normalize(image), "Y X -> () () Y X")
        _image = torch.Tensor(_image).to(self.device) 
        _image = self.quilter.unstitch(_image)

        with torch.no_grad():
            _result = self.model(_image)
        _result = nn.Softmax(dim=1)(_result).to('cpu')
        _result = self.quilter.stitch(_result)

        result = np.zeros((shape[0], shape[1]), dtype=float)
        #Note from Albert: result is M x C x H x W, where M can be > 1 if you're
        #stitching more than 1 image together. We are not, so
        #we index into 0
        result += np.array(_result[0][0,0]<0.9, dtype=float)
        return result

    def save(self, file='./model.pt'):
        torch.save(self.model, file)
        # save qulter???

    def load(self, file):
        self.model = torch.load(file, map_location=torch.device(self.device))
        # load quilter???

    def parse_dataset(self, dataset, include_data, batch_size, shuffle, drop_last):
        X, y = [], []
    
        for (i, images), (j, labels) in zip(dataset.images.items(), dataset.labels.items()):
            for k in include_data[i]:
                image = self.normalize(images[k])
                label = labels[k]
                X.append(image)
                y.append(label)

                # rotate 90
                image = np.rot90(image, k=1)
                label = np.rot90(label, k=1)
                X.append(image)
                y.append(label)

                # rotate 180
                image = np.rot90(image, k=1)
                label = np.rot90(label, k=1)
                X.append(image)
                y.append(label)

                # rotate 270
                image = np.rot90(image, k=1)
                label = np.rot90(label, k=1)
                X.append(image)
                y.append(label)

                # rotate 360
                image = np.rot90(image, k=1)
                label = np.rot90(label, k=1)

                # flip 0
                X.append(np.flip(image, 0))
                y.append(np.flip(label, 0))
                
                # flip 1
                X.append(np.flip(image, 1))
                y.append(np.flip(label, 1))

        X = einops.rearrange(X, "N Y X -> N () Y X")
        y = einops.rearrange(y, "N Y X -> N () Y X")
        X, y = torch.Tensor(X), torch.Tensor(y)

        X, y = self.quilter.unstitch_data_pair(X, y)

        S = len(X) * 80 // 100
        #print(S)
        #import sys; sys.exit()
        mean = torch.mean(y, dim=(1,2,3)).numpy()
        order = np.argsort(mean)
        X, y = X[order], y[order]

        sel = np.random.choice(X.shape[0], 2*S)
        selt = sel[:S]
        selv = sel[S:]
        tX, vX = X[selt], X[selv]
        ty, vy = y[selt], y[selv]

        tloader = DataLoader(TensorDataset(tX, ty),
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last)
        vloader = DataLoader(TensorDataset(vX, vy),
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last)

        return tloader, vloader
        
    def normalize(self, image):
        footprint = disk(32)
        img = np.log(np.abs(image) - np.min(image) + 1e-7)
        p2, p98 = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(p2, p98))
        img = skimage.util.img_as_ubyte(img)
        img_eq = rank.equalize(img, selem=footprint)
        img_eq = img_eq.astype(float)/256.0
        return img_eq       
