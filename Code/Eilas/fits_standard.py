# Make a standard size of an fits file. 
from matplotlib import pyplot as plt
from astropy.io import fits
import glob
from clustar.core import ClustarData
from astropy.visualization import ZScaleInterval

def fits2png(file_path):
    zscale = ZScaleInterval(contrast=0.25, nsamples=1)
    data = fits.getdata(file_path)
    data = zscale(data).squeeze()
    return data

# Find object position, fits-input
def find_object_pos(file):
    cd = ClustarData(path=file, group_factor=0)
    if len(cd.groups) > 0:
        disk = cd.groups[0]
        bounds = disk.image.bounds
        x = (bounds[2] + bounds[3])/2
        y = (bounds[0] + bounds[1])/2
        return (x, y)
    else:
        print("No object found in {}".format(file) + ", asuming middle")
        mid = fits.getdata(file).squeeze().shape[0]
        object_pos = (int(mid/2), int(mid/2))
        return object_pos 
    
# Crops image with object centered, image-input
def im_crop_object(file, object_pos, im_size):
    (x_mid, y_mid) = (int(object_pos[0]), int(object_pos[1]))
    (x_lower, x_upper) = (int(x_mid - (im_size)/2), int(x_mid + (im_size)/2))
    (y_lower, y_upper) = (int(y_mid - (im_size)/2), int(y_mid + (im_size)/2))
    im_crop = file[y_lower:y_upper, x_lower:x_upper]
    return im_crop

# Find file and open, path-input
def fits2standard(fits_data, im_size):
    obj_pos = find_object_pos(fits_data)
    data = fits2png(fits_data) 
    data = im_crop_object(data, obj_pos, im_size).reshape(1, 1, im_size, im_size)
    return data

def fits_folder2standard(fits_folder, im_size):
    filename = glob.glob(fits_folder +'/*.fits')
    for file in filename:
        data = fits2standard(file, im_size)
        return data