import cv2
import numpy as np
import os
import csv
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Function to apply augmentations
def augment_image(image):
    augmented_images = []
    # 1. Horizontal Flip
    augmented_images.append(cv2.flip(image, 1))
    # 2. Vertical Flip
    augmented_images.append(cv2.flip(image, 0))
    # 3. Rotation
    angle = random.uniform(-30, 30)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    augmented_images.append(cv2.warpAffine(image, M, (w, h)))
    # 4. Scaling
    scale = random.uniform(0.8, 1.2)
    augmented_images.append(cv2.resize(image, None, fx=scale, fy=scale))
    # 5. Brightness Adjustment
    bright_image = cv2.convertScaleAbs(image, alpha=random.uniform(0.8, 1.2), beta=0)
    augmented_images.append(bright_image)
    return augmented_images

# Blob detection function
def detect_blobs(image):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.2
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    return len(keypoints), keypoints

# Paths
input_folder = "name_date_images/train/year"
output_folder = 'object_detection_results/simple_blob_output_images'
augmented_folder = 'object_detection_results/simple_blob_augmented_images'
os.makedirs(augmented_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# CSV File
csv_file = "object_detection_results/simple_blob_results.csv"

# Initialize variables for accuracy calculation
total_images = 0
correct_matches = 0
expected_list = []
detected_list = []

# Open CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "Detected Circles", "Expected Circles", "Correct Match"])  # Header row

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(input_folder, filename)

            # Load image
            image = cv2.imread(filepath, 0)

            # Augment and save images
            augmented_images = augment_image(image)
            for i, aug_image in enumerate(augmented_images):
                aug_filename = f"Augmented_{i+1}_{filename}"
                aug_filepath = os.path.join(augmented_folder, aug_filename)
                cv2.imwrite(aug_filepath, aug_image)

                # Blob detection on augmented image
                detected_circles, keypoints = detect_blobs(aug_image)

                # Extract expected circles from filename
                match = re.search(r"year(\d+)", filename)
                expected_circles = int(match.group(1)) if match else 0

                # Save processed image with annotations
                output_image = cv2.drawKeypoints(aug_image, keypoints, np.zeros_like(aug_image), 
                                                 (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.putText(output_image, f"Detected: {detected_circles}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                output_path = os.path.join(output_folder, f"Processed_{aug_filename}")
                cv2.imwrite(output_path, output_image)

                # Record results
                correct_match = detected_circles == expected_circles
                correct_matches += int(correct_match)
                total_images += 1
                expected_list.append(expected_circles)
                detected_list.append(detected_circles)
                writer.writerow([aug_filename, detected_circles, expected_circles, correct_match])

# Calculate accuracy
accuracy = (correct_matches / total_images) * 100 if total_images > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")

# Generate confusion matrix
conf_matrix = confusion_matrix(expected_list, detected_list)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Detected Circles")
plt.ylabel("Expected Circles")
plt.savefig("confusion_matrix.png")
plt.show()
plt.savefig("simpleBlob_conf_mat.png")




# import cv2 
# import numpy as np 

# # Load image 
# image = cv2.imread('003-002-a-005-year12flint.png', 0) 

# # Set our filtering parameters 
# # Initialize parameter setting using cv2.SimpleBlobDetector 
# params = cv2.SimpleBlobDetector_Params() 

# # Set Area filtering parameters 
# params.filterByArea = True
# params.minArea = 30

# # Set Circularity filtering parameters 
# params.filterByCircularity = True
# params.minCircularity = 0.1

# # Set Convexity filtering parameters 
# params.filterByConvexity = True
# params.minConvexity = 0.2
	
# # Set inertia filtering parameters 
# params.filterByInertia = True
# params.minInertiaRatio = 0.1

# # Create a detector with the parameters 
# detector = cv2.SimpleBlobDetector_create(params) 
	
# # Detect blobs 
# keypoints = detector.detect(image) 

# # Draw blobs on our image as red circles 
# blank = np.zeros((1, 1)) 
# blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), 
# 						cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

# number_of_blobs = len(keypoints) 
# text = "Number of Circular Blobs: " + str(len(keypoints)) 
# cv2.putText(blobs, text, (20, 550), 
# 			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 

# print("Number of blobs", number_of_blobs)
# # Show blobs 
# # cv2.imshow("Filtering Circular Blobs Only", blobs) 
# cv2.imwrite("simpl_blob.png", blobs)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 
