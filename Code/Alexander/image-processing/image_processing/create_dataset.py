import os
from matplotlib import pyplot as plt
from astropy.nddata import Cutout2D
from astropy import units
from astropy.io import fits
from astropy.io import fits
from clustar.core import ClustarData
from astropy.visualization import ZScaleInterval
from astropy.io import fits
# import pandas as pd
# import csv
# import numpy as np
# from astropy.modeling.models import Gaussian2D
# import glob
# from astropy.visualization import astropy_mpl_style
# from astropy.utils.data import get_pkg_data_filename
# import random
# from scipy.ndimage import rotate


def file_fits_to_png(file_path, save_path, name):
    data = fits.getdata(file_path)
    zscale = ZScaleInterval()
    data = zscale(data).squeeze()
    plt.imshow(data)
    plt.axis('off')
    plt.ioff()
    plt.savefig('../data/output/create_dataset/' + name + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def data_fits_to_png(data, save_path, name):
    zscale = ZScaleInterval(contrast=0.25, nsamples=1)
    plt.figure(figsize=(image_size, image_size), dpi=100)
    plt.imshow(zscale(data.data).squeeze(), origin='lower', cmap='rainbow', aspect='auto')
    plt.axis('off')
    plt.ioff()
    plt.savefig('../data/output/create_dataset/' + name + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()


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


if __name__ == "__main__":
    train_pos = {os.path.splitext(f)[0] : 1 for f in os.listdir('../data/input/train/pos')}
    train_neg = {os.path.splitext(f)[0] : 0 for f in os.listdir('../data/input/train/neg')}

    eval_pos = {os.path.splitext(f)[0] : 1 for f in os.listdir('../data/input/eval/pos')}
    eval_neg = {os.path.splitext(f)[0] : 0 for f in os.listdir('../data/input/eval/neg')}

    dict_train = {**train_pos, **train_neg}
    dict_eval = {**eval_pos, **eval_neg}

    image_size = 100

    #Spara alla .fits filer som png för inmatninig i CNN
    for name, label in dict_train.items():

        print(name)
        print(label)

        if (label):
            img_data = fits.getdata('../data/input/train/pos/' + name + '.fits')
            object_pos = find_object_pos('../data/input/train/pos/' + name + '.fits')

        else:
            img_data = fits.getdata('../data/input/train/neg/' + name + '.fits')
            object_pos = find_object_pos('../data/input/train/neg/' + name + '.fits')

        if object_pos != None:
            # Data shape is (1, 1, x, y) we want it to be (x, y)
            img_data.shape = (img_data.shape[2], img_data.shape[3])
            # Set the size of the crop in pixels
            crop_size = units.Quantity((image_size, image_size), units.pixel)
            data = Cutout2D(img_data, object_pos, crop_size)

        if (label): data_fits_to_png(data, '../data/input/train/', name)

        else: data_fits_to_png(data, '../data/input/train/', name)

    for name, label in dict_eval.items():
        if (label): file_fits_to_png('../data/input/eval/pos/' + name + '.fits', '../data/input/eval/', name)
        else: file_fits_to_png('../data/input/eval/neg/' + name + '.fits', '../data/input/eval/', name)

    #Spara annotations för false och positives
    with open('../data/output/create_dataset/annotations.csv', 'w') as f:
        for key in dict_train.keys():
            f.write("%s %s\n"%(key,dict_train[key]))

    with open('../data/output/create_dataset/annotations.csv', 'w') as f:
        for key in dict_eval.keys():
            f.write("%s %s\n"%(key,dict_train[key]))

        
