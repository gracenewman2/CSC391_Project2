# Grace Newman SIFT Descriptors and Scaling

import cv2

# Read image
img = cv2.imread('C:/Users/ciezcm15/Documents/Project2/IMG_1293_rotate.jpg')
# img1 = cv2.imread('C:/Users/ciezcm15/Documents/Project2/IMG_1293.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# default parameters : 0, 3, 0.04, 10, 1.6
sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10, 1.6)
# sift1 = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10, 1.6)
kp = sift.detect(gray, None)
# kp1 = sift1.detect(gray1, None)

img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img1 = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("sift image", img)
# cv2.imshow("sift", img1)
key = cv2.waitKey(0)
cv2.imwrite("C:/Users/ciezcm15/Documents/Project2/siftImage_rotate.jpg", img)