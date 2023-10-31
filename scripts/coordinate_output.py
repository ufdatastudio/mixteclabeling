import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('045-a-01.png',cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('045.png',cv.IMREAD_GRAYSCALE) # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

if len(good) >= 4:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
    
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    h,w = img1.shape
    pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts,M)
    
    # Convert the grayscale image to a 3-channel BGR image
    img2_bgr = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    
    # Draw the red bounding box
    img2_bgr = cv.polylines(img2_bgr, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)

img3 = cv.drawMatches(img1, kp1, img2_bgr, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.show()
