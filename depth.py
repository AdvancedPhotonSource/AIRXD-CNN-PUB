import os
import math
import imageio
import numpy as np
import glob as glob
from time import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from airxd_cnn.cnn import ARIXD_CNN as cmodel
from airxd_cnn.dataset import Dataset
from sklearn.metrics import confusion_matrix as CM

import warnings
warnings.filterwarnings("ignore")

if not os.path.isdir('./pngs'):
    os.mkdir('./pngs')
if not os.path.isdir('./models'):
    os.mkdir('./models')

directories = ['./data/training/Battery_1',
               './data/training/Battery_2',
               './data/training/Battery_3',
               './data/training/Battery_4',
               './data/training/Nickel',
               ]
dataset = Dataset(n=len(directories))
dataset.get_data(directories, label_ext='.tif')

Ns = [128]
depths = [2]
for N in Ns:
    for depth in depths:
        epoch, batch_size = 200, 50
        
        print('N ', N)
        print('depth ', depth)
        print('epoch ', epoch)
        print('batch ', batch_size)

        # Quilter params
        M = N // 2
        B = M // 4
        quilter_params = {'Y': 2880, 'X': 2880,
                          'window': (N, N),
                          'step': (M, M),
                          'border': (B, B),
                          'border_weight': 0}

        # TUNet params
        model_params = {'image_shape': (2880, 2880),
                        'in_channels': 1,
                        'out_channels': 2,
                        'base_channels': 8,
                        'growth_rate': 2,
                        'depth': depth}
        
        model = cmodel(quilter_params, model_params, device='cuda:0')
        model.train(dataset, include_data={0: [0, 1, 2],
                                           1: [0, 1, 2],
                                           2: [0, 1, 2],
                                           3: [0, 1, 2],
                                           4: [0, 1, 2],
                                           5: [0, 1, 2]},
                    epoch=epoch,
                    batch_size=batch_size,
                    lr_rate=1e-2)
        model_name = f'./N{N}_depth{depth}_epoch{epoch}_batch{batch_size}.pt'
        model.save(model_name)

        # Predict
        datadirs = ["Battery_1", "Battery_2", "Battery_3", "Battery_4", "Nickel"]
        for _datadir in datadirs:
            print(f'========================= {_datadir} =========================\n')
            datadir = f"data/test/{_datadir}/"
            maskdir = f"data/test/{_datadir}/mask/"

            names = []
            for file in os.listdir(datadir):
                if 'tif' in file:
                    names.append(file[:-4])
            
            t = []
            mattol = np.zeros(4)
            for file in names:
                # Predict
                t0 = time()
                image = imageio.volread(datadir+file+".tif")
                label = np.array(imageio.volread(maskdir+file+"_mask.tif"), dtype=int) 
                label_pred = model.predict(image)
                t1 = time()
                t.append(t1-t0)

                matrix = CM(label.ravel(), label_pred.ravel())
                mattol += matrix.ravel()
                tn, fp, fn, tp = matrix.ravel()
                tn_rate = tn/(fp+tn)*100
                tp_rate = tp/(fn+tp)*100
                print(f'True Negative   : {tn}')
                print(f'False Positive  : {fp}')
                print(f'False Negative  : {fn}')
                print(f'True Positive   : {tp}')
                print(f'True TN rate    : {round(tn_rate,1)} %')
                print(f'True TP rate    : {round(tp_rate,1)} %')
                print('\n')
                
            t_avg = np.average(t)
            tn_rate = mattol[0]/(mattol[1]+mattol[0])*100
            tp_rate = mattol[3]/(mattol[2]+mattol[3])*100
            
            print(f'Overall prediction time     : {round(t_avg, 1)} seconds')
            print(f'Overall true negative rate  : {round(tn_rate, 1)} %') 
            print(f'Overall true positive rate  : {round(tp_rate, 1)} %\n')
