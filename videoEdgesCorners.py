# Grace Newman Edges and Corners in Video

import cv2
import numpy as np

# record from camera
cv2.namedWindow("preview")
vid = cv2.VideoCapture(0)

if vid.isOpened(): # try to get the first frame
    rval, frame = vid.read()
else:
    rval = False

# write the video to file
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# parameter is False when the video is in black & white (edges video) and True for color video (corners)
out = cv2.VideoWriter('C:/Users/ciezcm15/Documents/Project2/edges_dark.avi', fourcc, 25.0, (int(vid.get(3)), int(vid.get(4))), False)

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vid.read()
    grayVid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny edges on image
    edges = cv2.Canny(grayVid, 75, 50) # 100, 150
    # edges2 = cv2.Canny(grayVid, 100, 150)
    cv2.imshow("Canny Edge", edges)
    # cv2.imshow("edges", edges2)
    # out.write(edges)

    # Harris corners
    grayVid = np.float32(grayVid)
    dst = cv2.cornerHarris(grayVid, 3, 3, .2) #3, 3, 0.1
    dst = cv2.dilate(dst, None)
    frame[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow("Corner Harris", frame)
    # out.write(frame)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vid.release()
out.release()
cv2.destroyWindow("preview")



