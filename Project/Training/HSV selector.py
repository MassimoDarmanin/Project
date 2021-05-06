# import the necessary packages
import argparse
import time
from collections import deque

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

# www.colorizer.org

# https://online-image-comparison.com/

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
                help="max buffer size")
args = vars(ap.parse_args())


# if a video path was not supplied, grab the reference
# to the webcam
# if not args.get("video", False):
#    camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
# else:
#    camera = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
# time.sleep(2.0)


def nothing(x):
    pass


cv2.namedWindow('marking')

cv2.createTrackbar('H Lower', 'marking', 0, 179, nothing)
cv2.createTrackbar('H Higher', 'marking', 179, 179, nothing)
cv2.createTrackbar('S Lower', 'marking', 0, 255, nothing)
cv2.createTrackbar('S Higher', 'marking', 255, 255, nothing)
cv2.createTrackbar('V Lower', 'marking', 0, 255, nothing)
cv2.createTrackbar('V Higher', 'marking', 255, 255, nothing)

while (1):
    img = cv2.imread(args.get("image"))
    # img = cv2.flip(img, 1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hL = cv2.getTrackbarPos('H Lower', 'marking')
    hH = cv2.getTrackbarPos('H Higher', 'marking')
    sL = cv2.getTrackbarPos('S Lower', 'marking')
    sH = cv2.getTrackbarPos('S Higher', 'marking')
    vL = cv2.getTrackbarPos('V Lower', 'marking')
    vH = cv2.getTrackbarPos('V Higher', 'marking')

    LowerRegion = np.array([hL, sL, vL], np.uint8)
    upperRegion = np.array([hH, sH, vH], np.uint8)

    mask = cv2.inRange(hsv, LowerRegion, upperRegion)

    kernal = np.ones((1, 1), "uint8")

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    mask = cv2.dilate(mask, kernal, iterations=1)
    cv2.imshow("Mask ", mask)

    # compute masked original image
    res1 = cv2.bitwise_and(img, img, mask=mask)

    # compute area using black pixels in mask
    pixelCount = cv2.countNonZero(mask)
    outtext = 'Pixel Area: ' + str(pixelCount)

    # font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    # position
    pos = (10, 30)
    # fontScale
    fontScale = 1
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2

    cv2.putText(res1, outtext, pos, font, fontScale, color, thickness, cv2.LINE_AA)
    # print('Number of pixels', str(pixelCount))

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw all contours
    # image = cv2.drawContours(res1, contours, -1, (0, 255, 0), 2)

    print(len(contours))

    # print(contours)

    # show the image with the drawn contours

    # iterate through contours list
    count = 0
    for c in contours:
        area = cv2.contourArea(c)

        if area > 3:
            M = cv2.moments(c)
            # print(M)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            hull = cv2.convexHull(c)
            cv2.circle(res1, (cX, cY), 2, (255, 0, 0), -1)
            cv2.drawContours(res1, contours, 0, (255, 0, 0), 2)
            print('Area of contour', count, ' is ', area)
            count += 1

    print(count)

    cv2.imshow("Masking ", res1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        # camera.release()
        cv2.destroyAllWindows()
        break
