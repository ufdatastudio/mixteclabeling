import os
import random
from PIL import Image
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Augment images by performing random rotations.')
parser.add_argument('image_directory', type=str, help='Directory containing the subdirectories of images to augment')

# Parse the command-line arguments
args = parser.parse_args()

# Define the directory containing the images
image_directory = args.image_directory

# Define the range of rotation angles
rotation_angles = range(15, 360, 15)  # Rotate by multiples of 15 degrees

# Iterate over each subdirectory and file in the image directory
for root, dirs, files in os.walk(image_directory):
    for filename in files:
        if filename.endswith('.png'):  # Check for image files
            image_path = os.path.join(root, filename)
            
            # Open the image
            with Image.open(image_path) as img:
                # Perform random rotations
                for angle in random.sample(rotation_angles, k=8):  # Choose 8 random angles
                    rotated_img = img.rotate(angle, expand=True)
                    
                    # Save the augmented image with a new filename
                    augmented_filename = f'aug_{angle}_{filename}'
                    augmented_img_path = os.path.join(root, augmented_filename)
                    rotated_img.save(augmented_img_path)

print("Augmentation completed successfully!")
