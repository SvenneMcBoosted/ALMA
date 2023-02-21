import joblib
from skimage.transform import resize
from skimage.io import imread
from init_modules import *


# Set the path to the data directory
filenames = glob.glob('data/*.fits')
data_set = {}

im_size = 100
shift_interval = 1
gaussians = []


# This function finds the position of the object in the image
def find_object_pos(file):
    cd = ClustarData(path=file, group_factor=0)
    if len(cd.groups) > 0:
        disk = cd.groups[0]
        bounds = disk.image.bounds
        x = (bounds[2] + bounds[3])/2
        y = (bounds[0] + bounds[1])/2
        print('alwkfnlawnflkn')
        return (x, y)
    else:
        print("No object found in {}".format(file))
        return None


# This function crops the image to the size of the object
def init_cropped_images():
    for file in filenames:
        img_data = fits.getdata(file)
        object_pos = find_object_pos(file)

        if object_pos != None:
            # Data shape is (1, 1, x, y) we want it to be (x, y)
            img_data.shape = (img_data.shape[2], img_data.shape[3])

            # Set the size of the crop in pixels
            crop_size = units.Quantity((im_size, im_size), units.pixel)

            img_crop = Cutout2D(img_data, object_pos, crop_size)

            gaussians.append(img_crop)


# This function rotates the image by a random angle
def rotate_disk(disk_to_rotate, angle):

    # Rotate the disk
    rotated_disk = rotate(disk_to_rotate, angle)
    # Since rotating pads the image, we need to crop it to the original size
    x, y = (len(rotated_disk[0]), len(rotated_disk))

    shift_interval = 8
    si = shift_interval + 1

    rand_x_shift = random.randint(-shift_interval, shift_interval)
    rand_y_shift = random.randint(-shift_interval, shift_interval)

    (x_lower, x_upper) = int((x/2 - im_size/2)) + \
        rand_x_shift, int(x/2 + im_size/2) + rand_x_shift
    (y_lower, y_upper) = int((y/2 - im_size/2)) + \
        rand_y_shift, int(y/2 + im_size/2) + rand_y_shift

    return rotated_disk[(x_lower+si):(x_upper-si), (y_lower+si):(y_upper-si)]


# This function flips the image horizontally, vertically or both
def flip_disk(disk_to_flip):

    flipped_disk = disk_to_flip

    if bool(random.getrandbits(1)):
        flipped_disk = np.fliplr(flipped_disk)

    if bool(random.getrandbits(1)):
        flipped_disk = np.flipud(flipped_disk)

    if bool(random.getrandbits(1)):
        flipped_disk = np.flip(flipped_disk)

    return flipped_disk


# This function augments the image by rotating and flipping it
def augment_disk(disk):
    angle = random.randint(0, 360)
    return rotate_disk(flip_disk(disk), angle)


# This function generates the positive dataset
def generate_pos_dataset(augmentations_per_gaussian):
    count = 0
    pos_dataset = []
    for gaussian in gaussians:
        for i in range(0, augmentations_per_gaussian):
            zscale = ZScaleInterval(contrast=0.25, nsamples=1)
            # Augment the data and add it to the dataset as png
            plt.figure()
            plt.imshow(zscale(augment_disk(gaussian.data)),
                       origin='lower', cmap='rainbow')
            # plt.colorbar()
            # plt.title("Augmented gaussian")
            # plt.show()

            # pos_dataset.append(zscale(augment_disk(gaussian.data)))
    return pos_dataset


# This function generates the negative dataset

def generate_neg_dataset(augmentations_per_gaussian):
    neg_dataset = []
    im_center = im_size/2
    y, x = np.mgrid[0:im_size, 0:im_size]
    for i in range(0, len(filenames)):
        for j in range(0, augmentations_per_gaussian):
            rand_x_shift = random.randint(-shift_interval, shift_interval)
            rand_y_shift = random.randint(-shift_interval, shift_interval)
            data = Gaussian2D(1, im_center + rand_x_shift,
                              im_center +
                              rand_y_shift, random.randrange(5, 20),
                              random.randrange(5, 20), theta=random.randrange(0, 2))(x, y)

            zscale = ZScaleInterval(contrast=0.25, nsamples=1)
            neg_dataset.append(zscale(augment_disk(data)))
    return neg_dataset


def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary,
    together with labels and metadata. The dictionary is written to a pickle file
    named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """

    height = height if height is not None else width

    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(
        int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    pklname = f"{pklname}_{width}x{height}px.pkl"

    for image in src:
        im = imread(os.path.join(current_path, file))
        im = image  # [:,:,::-1]
        data['label'].append(subdir[:-4])
        data['filename'].append(file)
        data['data'].append(im)

    joblib.dump(data, pklname)


if __name__ == '__main__':

    X_test = generate_pos_dataset(50)
    y_test = generate_neg_dataset(50)

    print(X_test.shape)

    plt.figure()
    plt.imshow(X_test[0], origin='lower', cmap='rainbow')
    plt.colorbar()
    plt.title("Augmented gaussian")
    plt.show()

    X_train = generate_pos_dataset(50)
    y_train = generate_neg_dataset(50)

    # Save X_test and y_test
    np.save('X_test', X_test)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     x, y, test_size=.33)
