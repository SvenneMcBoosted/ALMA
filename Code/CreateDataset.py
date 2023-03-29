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
import random
from scipy.ndimage import rotate

def fits2pngFile(file_path, save_path, name):
    data = fits.getdata(file_path)
    zscale = ZScaleInterval()
    data = zscale(data).squeeze()
    plt.imshow(data)
    plt.axis('off')
    plt.ioff()
    plt.savefig('./data/train/' + name + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def fits2pngData(data, save_path, name):
    zscale = ZScaleInterval(contrast=0.25, nsamples=1)
    plt.figure(figsize=(im_size, im_size), dpi=100)
    plt.imshow(zscale(data.data).squeeze(), origin='lower', cmap='rainbow', aspect='auto')
    plt.axis('off')
    plt.ioff()
    plt.savefig('./data/train/' + name + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def saveFits(data, save_path, name):
    n = np.arange(100.0)
    hdu = fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    hdul.data = data.data
    hdul.writeto(save_path + name + '.fits')


train_positives = {os.path.splitext(f)[0] : 1 for f in os.listdir('./data/org/fits/pos/')}
train_negatives = {os.path.splitext(f)[0] : 0 for f in os.listdir('./data/org/fits/neg/')}
eval_positives = {os.path.splitext(f)[0] : 1 for f in os.listdir('./data/eval/pos/')}
eval_negatives = {os.path.splitext(f)[0] : 0 for f in os.listdir('./data/eval/neg/')}
train_dt = {**train_positives, **train_negatives}
eval_dt = {**eval_positives, **eval_negatives}

#augmentera här, lägg till .fits kopior i train_dt

im_size = 100

def find_object_pos(file):
    cd = ClustarData(path=file, group_factor=0, threshold=0.025, metric="standard_deviation")
    if len(cd.groups) > 0:
        disk = cd.groups[0]
        bounds = disk.image.bounds
        x = (bounds[2] + bounds[3])/2 
        y = (bounds[0] + bounds[1])/2
        return (x, y)
    else:
        print("No object found in {}".format(file))
        return None


#Spara alla .fits filer som png för inmatninig i CNN
for name, label in train_dt.items():
    if (label):
        img_data = fits.getdata('./data/org/fits/pos/' + name + '.fits')
        object_pos = find_object_pos('./data/org/fits/pos/' + name + '.fits')
    else:
        img_data = fits.getdata('./data/org/fits/neg/' + name + '.fits')
        object_pos = find_object_pos('./data/org/fits/neg/' + name + '.fits')
    
    if object_pos != None:
        # Data shape is (1, 1, x, y) we want it to be (x, y)
        img_data.shape = (img_data.shape[2], img_data.shape[3])
        # Set the size of the crop in pixels
        crop_size = units.Quantity((im_size, im_size), units.pixel)
        data = Cutout2D(img_data, object_pos, crop_size)
    if (label): saveFits(data, './data/train/', name)
    else: saveFits(data, './data/train/', name)
    

for name, label in eval_dt.items():
    if (label): fits2pngFile('./data/eval/pos/' + name + '.fits', './data/eval/', name)
    else: fits2pngFile('./data/eval/neg/' + name + '.fits', './data/eval/', name)

#Spara annotations för false och positives
with open('./data/train/annotations.csv', 'w') as f:
    for name, label in train_dt.items():
        f.write("%s %s\n"%(name,label))

with open('./data/eval/annotations.csv', 'w') as f:
    for name, label in train_dt.items():
        f.write("%s %s\n"%(name,label))


