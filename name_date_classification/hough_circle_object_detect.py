import cv2
import numpy as np
import os
import csv
import re
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Function to apply random data augmentations to the images
def augment_image(image):
    augmented_images = []
    
    # 1. Horizontal Flip
    flipped_h = cv2.flip(image, 1)
    augmented_images.append(flipped_h)
    
    # 2. Vertical Flip
    flipped_v = cv2.flip(image, 0)
    augmented_images.append(flipped_v)
    
    # 3. Random Rotation (between -30 and 30 degrees)
    angle = random.uniform(-30, 30)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented_images.append(rotated)
    
    # 4. Random Scaling (0.8x to 1.2x)
    scale = random.uniform(0.8, 1.2)
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    augmented_images.append(resized)
    
    # 5. Random Brightness Adjustment
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float32)
    random_brightness = random.uniform(0.5, 1.5)
    hsv[:, :, 2] = hsv[:, :, 2] * random_brightness
    hsv = np.array(hsv, dtype=np.uint8)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(bright)
    
    return augmented_images

# Read image and process
def detect_circles(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not open or find the image: {image_path}")
        return None, 0

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur using 3x3 kernel
    gray_blurred = cv2.blur(gray, (3, 3))
    
    # Apply Hough Transform on the blurred image
    detected_circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=0.5, 
        minDist=16, 
        param1=20, 
        param2=27, 
        minRadius=2, 
        maxRadius=17
    )
    
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        
        # Draw detected circles
        for circle in detected_circles[0, :]:
            a, b, r = circle  # (x, y, radius)
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)  # Draw the circle
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)  # Draw the center of the circle
        
        return img, len(detected_circles[0])  # Return image and number of circles detected
    else:
        return img, 0

# Directory paths
input_folder = 'name_date_images/train/year'
output_folder = 'object_detection_results/hough_circle_output_images'
augmented_folder = 'object_detection_results/hough_circle_augmented_images'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(augmented_folder, exist_ok=True)

# CSV file to save results
csv_file = "object_detection_results/hough_circle_results.csv"

# Initialize variables for accuracy calculation
total_images = 0
correct_detections = 0
expected_counts = []
detected_counts = []

# Open CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "Detected Circles", "Expected Circles", "Correct Match"])  # Header row

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"Detected_{filename}")
            
            # Extract the expected number of circles from the filename using regex
            try:
                match = re.search(r"year(\d+)", filename)
                if match:
                    expected_circles = int(match.group(1))  # Extract and convert to integer
                else:
                    expected_circles = -1  # Default value if "year" followed by digits is not found
            except Exception as e:
                expected_circles = -1
                
            
            # Detect circles in the original image
            processed_img, detected_circles = detect_circles(input_path)
            
            # Save the processed image
            if processed_img is not None:
                cv2.imwrite(output_path, processed_img)
                
            # Track expected and detected counts for confusion matrix
            expected_counts.append(expected_circles)
            detected_counts.append(detected_circles)
            
            # Check if the detected count matches the expected count
            correct_match = detected_circles == expected_circles
            if correct_match:
                correct_detections += 1
            total_images += 1
            
            # Write results to CSV
            writer.writerow([filename, detected_circles, expected_circles, correct_match])
            
            # Apply augmentations to the image and save augmented images
            augmented_images = augment_image(processed_img)
            for i, aug_img in enumerate(augmented_images):
                aug_output_path = os.path.join(augmented_folder, f"Augmented_{i+1}_{filename}")
                cv2.imwrite(aug_output_path, aug_img)
            
# Calculate and print accuracy percentage
accuracy_percentage = (correct_detections / total_images) * 100 if total_images > 0 else 0
print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")

conf_matrix = confusion_matrix(expected_counts, detected_counts)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", xticklabels=set(expected_counts), yticklabels=set(expected_counts))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Count")
plt.ylabel("True Count")
plt.show()
plt.savefig("conf_matrix.png")


