import pandas as pd
import csv
import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.nddata import Cutout2D
from astropy import units
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
import glob
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from clustar.core import ClustarData
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy.io.fits import getdata
from astropy.io.fits import getheader
from astropy.io.fits import getval
import random
from scipy.ndimage import rotate
from scipy import interpolate

def ShowFits(fitsPath):
    data = fits.getdata(fitsPath)
    print(data.shape)
    zscale = ZScaleInterval(contrast=0.25, nsamples=1)
    #plt.figure(figsize=(100, 100), dpi=100)
    plt.imshow(zscale(data.data).squeeze(), origin='lower', cmap='rainbow', aspect='auto')
    plt.show()
    plt.axis('off')
    plt.ioff()
    plt.savefig('./data/' + "test" + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def Zoomin(fitsPath, fileName, radius, center):
    hdul = fits.open(fitsPath + fileName, memmap=False)
    data = hdul[0].data
    hdul.close()
    matrix = data[0][0]
    mean = 0
    mini = 1e11
    maxi = -1e11
    for i in range(data.shape[2]):
        for j in range(data.shape[3]):
            mean += matrix[i][j]
            mini = min(mini, matrix[i][j])
            maxi = max(maxi, matrix[i][j])
    mean /= (data.shape[2]*data.shape[3])

    radius = min(radius, center[0], 100 - center[0], center[1], 100 - center[1])

    focus = matrix[center[1] - radius : center[1] + radius, center[0] - radius : center[0] + radius]
    print(focus.shape[0],focus.shape[1])
    x = np.linspace(mini,maxi,radius*2)
    y = np.linspace(mini,maxi,radius*2)
    f = interpolate.interp2d(x,y,focus,kind='cubic')
    x2 = np.linspace(mini, maxi, 100)
    y2 = np.linspace(mini, maxi, 100)
    arr2 = f(y2, x2)
    arr2.shape = (1,1,arr2.shape[0],arr2.shape[1])
    hdu = fits.PrimaryHDU(arr2)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fitsPath + fileName,overwrite=True)
    
    

    

#ChangeContrast("./data/train/b335_2017_band6_0.fits")

Zoomin("./data/train/", "b335_2017_band6_0.fits", 15, (10,10))
ShowFits("./data/train/b335_2017_band6_0.fits")
#ShowFits("./data/org/fits/pos/b335_2017_band6_0.fits")
