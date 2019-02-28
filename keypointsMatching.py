# Grace Newman Keypoints and Matching

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('C:/Users/ciezcm15/Documents/Project2/IMG_1293.jpg') # Image
img2 = cv2.imread('C:/Users/ciezcm15/Documents/Project2/IMG_1293_me.jpg') # small Image

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10, 1.6)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = None

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img3, flags=2)
cv2.imshow('MatchesKnn', img3)
key = cv2.waitKey(0)

cv2.imwrite("C:/Users/ciezcm15/Documents/Project2/different.jpg", img3)
