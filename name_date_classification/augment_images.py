import os
import random
from PIL import Image, ImageEnhance,ImageDraw
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Augment images by performing random rotations and color jitter.')
parser.add_argument('image_directory', type=str, help='Directory containing the subdirectories of images to augment')

# Parse the command-line arguments
args = parser.parse_args()

# Define the directory containing the images
image_directory = args.image_directory

# Define the range of rotation angles
rotation_angles = range(15, 360, 15)  # Rotate by multiples of 15 degrees

# Function to apply color jitter
def apply_color_jitter(img):
    # Randomly adjust brightness, contrast, and color
    brightness = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    contrast = ImageEnhance.Contrast(brightness).enhance(random.uniform(0.8, 1.2))
    color = ImageEnhance.Color(contrast).enhance(random.uniform(0, 2))
    return color

# Function to apply random masking
def apply_random_mask(img):
    # Convert to editable format
    img_with_mask = img.copy()
    draw = ImageDraw.Draw(img_with_mask)
    
    # Generate random dimensions for the mask
    width, height = img.size
    mask_width = random.randint(width // 8, width // 4)  # Mask size between 1/8 and 1/4 of the image width
    mask_height = random.randint(height // 8, height // 4)  # Mask size between 1/8 and 1/4 of the image height
    
    # Generate random position for the mask
    top_left_x = random.randint(0, width - mask_width)
    top_left_y = random.randint(0, height - mask_height)
    
    # Draw the mask (black rectangle)
    draw.rectangle(
        [(top_left_x, top_left_y), (top_left_x + mask_width, top_left_y + mask_height)],
        fill="white"
    )
    
    return img_with_mask


print("Data Augmentation: ⏎ Applying random rotation to images.")
# Step 1: Perform random rotations and save the rotated images
for root, dirs, files in os.walk(image_directory):
    for filename in files:
        if filename.endswith('.png'):  # Check for image files
            image_path = os.path.join(root, filename)
            
            # Open the image
            with Image.open(image_path) as img:
                # Perform random rotations
                for angle in random.sample(rotation_angles, k=16):  # Choose 16 random angles
                    rotated_img = img.rotate(angle, expand=True)
                    
                    # Save the rotated image with a new filename
                    rotated_filename = f'rot_{angle}_{filename}'
                    rotated_img_path = os.path.join(root, rotated_filename)
                    rotated_img.save(rotated_img_path)

print("Data Augmentation: 🎨 Applying random color jitter to images.")
# Step 2: Apply color jitter to each rotated image
for root, dirs, files in os.walk(image_directory):
    for filename in files:
        if  filename.endswith('.png'):  # Only process rotated images
            image_path = os.path.join(root, filename)
            
            # Open the rotated image
            with Image.open(image_path) as img:
                # Apply color jitter
                jittered_img = apply_color_jitter(img)
                
                # Save the color-jittered image with a new filename
                jittered_filename = f'jitter_{filename}'
                jittered_img_path = os.path.join(root, jittered_filename)
                jittered_img.save(jittered_img_path)

# Step 3: Apply random masking to each color-jittered image
print("Data Augmentation: 🕶️ Applying random masking to images.")
for root, dirs, files in os.walk(image_directory):
    for filename in files:
        if filename.endswith('.png'):  # Only process jittered images
            image_path = os.path.join(root, filename)
            
            # Open the jittered image
            with Image.open(image_path) as img:
                # Apply random mask
                masked_img = apply_random_mask(img)
                
                # Save the masked image with a new filename
                masked_filename = f'mask_{filename}'
                masked_img_path = os.path.join(root, masked_filename)
                masked_img.save(masked_img_path)

print("Data Augmentation: Rotation, color jitter, and masking augmentation completed successfully!")