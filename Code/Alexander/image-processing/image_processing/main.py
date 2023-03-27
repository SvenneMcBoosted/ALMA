import os
from astropy.io import fits
from astropy.nddata import Cutout2D
from clustar.core import ClustarData
import matplotlib.pyplot as plt


# Define the desired output image size
size = (100, 100)


def find_celestial_bodies(file_path):

    hdulist = fits.open(file_path)
    data = hdulist[0].data[0][0]  # Remove first two dimensions for (1, 1, 480, 480) shaped data

    cd = ClustarData(
        path=file_path,
        group_factor=0,
        threshold=0.025,
        metric="standard_deviation")  # 0.025 standard

    if len(cd.groups) > 0:

        disk = cd.groups[0]
        bounds = disk.image.bounds
        x = (bounds[2] + bounds[3]) / 2
        y = (bounds[0] + bounds[1]) / 2

        position = (x, y)
        cutout = Cutout2D(data, position, size)

        return cutout.data, True

    else:

        position = ((data.shape)[0] // 2, (data.shape)[0] // 2)
        cutout = Cutout2D(data, position, size)

        return cutout.data, False


def plot_and_save(image, file_name, output_dir, method_str):

    # Display the cropped image
    colormap = "inferno"
    # title = f"{file_name}_{colormap}_{method_str}"
    plt.imshow(image, origin="lower", cmap=colormap)

    plt.axis('off')
    plt.ioff()
    # plt.title(title)
    plt.savefig(output_dir + file_name + "_" + method_str, bbox_inches="tight", pad_inches=0)
    # plt.show()


# def get_fits_files(fits_directory):

#     # Initialize an empty list to store the file paths
#     file_paths = []

#     # Loop through all the files in the directory
#     for filename in os.listdir(fits_directory):
#         # Get the file path by joining the directory path and the filename
#         file_path = os.path.join(fits_directory, filename)
#         # Append the file path to the list
#         file_paths.append(file_path)

#     return file_paths

def get_fits_files(directory):
    """
    Returns a list of file paths for all FITS files in the given directory.
    """
    fits_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.fits'):
            fits_files.append(os.path.join(directory, filename))
    return fits_files


if __name__ == "__main__":

    output_dir = './app_data/output/'
    for file_path in get_fits_files('./app_data/input/'):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        cropped_image, method_bool = find_celestial_bodies(file_path)

        if method_bool:
            print(f"Found celestial body in {file_path}")
            method_str = "CLUSTAR"
        else:
            print(f"No object found in {file_path}")
            print("Assuming the object is centered")
            method_str = "MIDPOINT"

        plot_and_save(cropped_image, file_name, output_dir, method_str)

