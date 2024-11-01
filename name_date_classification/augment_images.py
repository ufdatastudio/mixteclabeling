import os
import random
from PIL import Image
import argparse


# Set up argument parsing
parser = argparse.ArgumentParser(description='Augment images by performing random rotations.')
parser.add_argument('image_directory', type=str, help='Directory containing the images to augment')

# Parse the command-line arguments
args = parser.parse_args()

# Define the directory containing the images
image_directory = args.image_directory


# Define the range of rotation angles
rotation_angles = range(0, 360, 15)  # Rotate by multiples of 15 degrees

# Iterate over each file in the image directory
for filename in os.listdir(image_directory):
    if filename.endswith('.png'):  # Check for image files
        image_path = os.path.join(image_directory, filename)
        
        # Open the image
        with Image.open(image_path) as img:
            # Perform random rotations
            for angle in random.sample(rotation_angles, k=3):  # Choose 3 random angles
                rotated_img = img.rotate(angle, expand=True)
                
                # Save the augmented image with a new filename
                augmented_filename = f'aug_{angle}_{filename}'
                augmented_img_path = os.path.join(image_directory, augmented_filename)
                rotated_img.save(augmented_img_path)

print("Augmentation completed successfully!")
