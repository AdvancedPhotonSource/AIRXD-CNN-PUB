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

if not os.path.isdir('./pngs'):
    os.mkdir('./pngs')

directories = ['./data/training/Battery_1',
               './data/training/Battery_2',
               './data/training/Battery_3',
               './data/training/Battery_4',
               #'./data/training/Battery_5',
               './data/training/Nickel',
               ]
dataset = Dataset(n=len(directories))
dataset.get_data(directories, label_ext='.tif')

# Quilter params
N = 256
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
                'depth': 4}

# Training params
epoch = 10
batch_size = 50
lr_rate = 1e-2

# Training
model = cmodel(quilter_params, model_params, device='cuda:0')
model.train(dataset, include_data={0: [0],
                                   1: [0],
                                   2: [0],
                                   3: [0],
                                   4: [0],
                                   5: [0]},
            epoch=epoch,
            batch_size=batch_size,
            lr_rate=lr_rate)
model.save('./test_model.pt')
# model.load('./test_model.pt')

# Predict and plot
datadirs = ["Battery_1", "Battery_2", "Battery_3", "Battery_4", "Battery_5", "Nickel"]
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
        
        # plot
        _max = np.max(image/30)
        log = int(math.log(_max, 10))
        maxx = _max / (log-3+0.1) / 10
        
        plt.figure(figsize=(20,10))
        mask_ori = np.ma.masked_where(label==0., label)
        plt.subplot(1, 2, 1)
        plt.title('Manual')
        plt.axis('off')
        plt.imshow(image, vmin=0, vmax=maxx, cmap='binary', origin='lower')
        plt.imshow(mask_ori, cmap=cm.autumn, origin='lower', interpolation='nearest')

        mask_pred = label_pred.astype(int)
        mask_pred = np.ma.masked_where(mask_pred==0., mask_pred)
        plt.subplot(1, 2, 2)
        plt.title('Prediction')
        plt.axis('off')
        plt.imshow(image, vmin=0, vmax=maxx, cmap='binary', origin='lower')
        plt.imshow(mask_pred, cmap=cm.autumn, origin='lower', interpolation='nearest')
        plt.tight_layout()
        plt.savefig(f'pngs/{_datadir}_{file}.png', dpi=300)
        plt.clf()
    
    t_avg = np.average(t)
    tn_rate = mattol[0]/(mattol[1]+mattol[0])*100
    tp_rate = mattol[3]/(mattol[2]+mattol[3])*100
    
    print(f'Overall prediction time     : {round(t_avg, 1)} seconds')
    print(f'Overall true negative rate  : {round(tn_rate, 1)} %') 
    print(f'Overall true positive rate  : {round(tp_rate, 1)} %\n')
