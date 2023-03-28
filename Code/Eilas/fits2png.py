# Functions for converting and saving fits-files as png 
from matplotlib import pyplot as plt
import glob
from astropy.io import fits
from astropy.visualization import ZScaleInterval


# Open fits-data and remove extra dimensions
def fits2png(file_path):
    zscale = ZScaleInterval(contrast=0.25, nsamples=1)
    data = fits.getdata(file_path)
    data = zscale(data).squeeze()
    return data

# Save data as an png in 
def save_png(im_data, name, folder_name):
    plt.imshow(im_data, origin='lower', cmap='rainbow', aspect='auto')
    plt.axis('off')
    plt.imsave(folder_name + '/'+ name + '.png',im_data, cmap='rainbow')

# Saves a folder of fits-files as png images
def fits_folder2png(fits_folder, png_folder):
    filenames = glob.glob(fits_folder + '/*fits')
    for file in filenames:
        im_data = fits2png(file)
        im_name = file.strip('.fits').strip(fits_folder)[1:]
        save_png(im_data, im_name, png_folder)

# Save image data as a fits-files
def fits_save(im_data, name, fits_folder):
    im_data.reshape((1,1,len(im_data.shape), len(im_data.shape)))
    hdu = fits.PrimaryHDU(im_data)
    hdu.writeto(fits_folder + name, overwrite=True)

