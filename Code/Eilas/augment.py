# Functions for augmentions of images
from fits2png import *
from fits_standard import *

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import glob
from astropy.visualization import PercentileInterval, ImageNormalize
from scipy.ndimage import rotate
import random
from astropy.modeling import models, fitting


# Rotates the image randomly, image-input
def im_rotate(im_to_rotate):
    rotated_disk = rotate(im_to_rotate, random.randint(-180, 180))
    return rotated_disk

# Takes the square of the image
def im_squere(im_file):
    squared_im = np.square(im_file)
    return  squared_im

# Takes the square root of the iamge, image input
def im_sqrt(im_file):
    abs_sqrt = np.sqrt(np.abs(im_file))
    sqrt_im = np.multiply(np.sign(im_file), abs_sqrt)
    return sqrt_im

# Geometric mean of images, image input
def im_geo_mean(image1, image2):
    im_geo = im_sqrt(image1 * image2)
    return im_geo


# Calculate angle of disk, fits-input
def disk_angle(fits_data):
    im_data = fits_data.squeeze()
    gaussian_model = models.Gaussian2D()
    fitter = fitting.LevMarLSQFitter()
    y, x = np.mgrid[:im_data.shape[0], :im_data.shape[1]]
    fit_model = fitter(gaussian_model, x, y, im_data)
    theta = np.arctan2(fit_model.y_stddev.value, fit_model.x_stddev.value)
    return theta

# Calculate the ellipcity of an disk, fits-input
def disk_ellipcity(fits_data):
    im_data = fits_data.squeeze()
    gaussian_model = models.Gaussian2D()
    fitter = fitting.LevMarLSQFitter()
    y, x = np.mgrid[:im_data.shape[0], :im_data.shape[1]]
    fit_model = fitter(gaussian_model, x, y, im_data)
    ellips = 1 - (fit_model.y_stddev / fit_model.x_stddev)
    return ellips

# Save calculated geo mean as a png, fits-input
def save_geo_mean(fits_data1, fits_data2, im_size, png_folder):
    data1 = fits2standard(fits_data1, im_size).squeeze()
    data2 = fits2standard(fits_data2, im_size).squeeze()
    im_geo = im_geo_mean(data1, data2)
    im_name1 = fits_data1.strip('.fits')[6:]
    im_name2 = fits_data2.strip('.fits')[6:]
    save_png(im_geo, im_name1+im_name2, png_folder)


#### INTE KLARA GREJER ####
# Moves the object randomly, image-input : Not done
def im_move(im_to_move):
    (x_size, y_size) = (len(im_to_move[0]), len(im_to_move))
    x_shift = random.randint(int(-x_size * 0.5) , int(x_size * 0.5))
    y_shift = random.randint(int(-y_size * 0.5), int(y_size * 0.5))
    print(x_shift, y_shift)
    (x_pos, y_pos) = find_object_pos(im_to_move)
    return im_crop_object(im_to_move, (x_pos + x_shift, y_pos + y_shift), 100)
def rotated_geo_mean(im_data1, im_data2):
    pass

# Calculate the noise level in the image
def im_noise(fits_data, im_size):
    im_data = fits2standard(fits_data, im_size)
    background_region = im_data[0:25, 0:25]
    mean, median, std = sigma_clipped_stats(background_region, sigma=3.0)
    return mean, median, std

def im_contrast(fits_data, im_size, lower_percentile, upper_percentile):
    im_data = fits2standard(fits_data, im_size).squeeze()
    interval = PercentileInterval(lower_percentile, upper_percentile)
    normalize = ImageNormalize(vmin=interval.get_limits(im_data)[0], vmax=interval.get_limits(im_data)[1])
    contrast_image = normalize(interval(im_data)) 
    return contrast_image

def im_noise_orginal(fits_data):
    im_data = fits.getdata(fits_data).squeeze()
    background_region = im_data[50:300, 50:300]
    mean, median, std = sigma_clipped_stats(background_region, sigma=3.0)
    return mean, median, std




