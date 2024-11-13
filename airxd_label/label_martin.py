import numpy as np
import numpy.ma as ma
from cffi import FFI
import scipy.special as sc
from glob import glob
import os
import sys

from airxd.mask import MASK

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    import imageio as iio
    from airxd_cnn.poni_parse import convert_to_imctrl
    from airxd_cnn.transforms import powder_normalize

    #Poni to Fit2D imctrl
    poni_path = 'data/from_martin/Q_MINES_JuneJuly2023_RC09_RawData/'
    numChans = 500
    #Fix the IOTTH problem described in autolabel_martin_data.ipnyb

    controls = convert_to_imctrl('data/from_martin/RC08/Q_MINES_JuneJuly2023_RC08_RawData/Calibration/LaB6_calibration_225mm_30keV.poni')
    controls['IOtth'] = [65.2, 72.95]
    
    #Import image
    example_im_dir = 'data/from_martin/Q_MINES_JuneJuly2023_RC09_RawData/'
    example_im_files = glob(example_im_dir + '*.Tiff')
    example_im = iio.v2.volread(example_im_files[0])

    #Try masking
    print('Normalizing image...')
    example_im = powder_normalize(example_im)



    mask = MASK(controls, shape= example_im.shape)
    result = mask.AutoSpotMask(example_im, esdmul = 2.0, numchans=numChans)
    np.save('mask_test', result)
