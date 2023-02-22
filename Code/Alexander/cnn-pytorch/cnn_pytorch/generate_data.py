import os
import torch
import torchvision.transforms as transforms
import numpy
import torchvision
from PIL import Image

# Define the input and output directories
input_dir = '../data/sets/input/'
output_dir = '../data/sets/output/'

# Define the list of transformations to be applied on images
transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0)),
    transforms.ToTensor(),
])

# Loop over all the images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # Load the image from disk and convert to RGB mode
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGB')
        
        # Apply the transformations to the image
        img_transformed = transformations(img)
        
        # Save the transformed image to the output directory
        output_filename = os.path.splitext(filename)[0] + '_transformed.png'
        output_path = os.path.join(output_dir, output_filename)
        torchvision.utils.save_image(img_transformed, output_path)
