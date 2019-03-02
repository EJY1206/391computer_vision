import cv2
import numpy as np


def print_howto():
    print("""
        Change different modes of image:
            1. Use Canny Edge Detection - press 'e'
            2. Use Harris Corner Detection - press 'c'
            3. Use SIFT Descriptors - press 's'
            4. Quit - press 'q'
    """)


def edge_image(img):

    # Otsu's thresholding
    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh

    edge = cv2.Canny(img, low_thresh, high_thresh)
    return edge


def corner_image(img, gray_img):

    dst = cv2.cornerHarris(gray_img, blockSize=4, ksize=5, k=0.04)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.02 * dst.max()] = [0, 0, 255]
    return img


def sift_image(img, gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(gray_img, None)

    cv2.drawKeypoints(img, keypoints, img, \
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img


print_howto()
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cur_mode = None
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

    if c != -1 and c != 255 and c != cur_mode:
        cur_mode = c

    if cur_mode == ord('e'):
        cv2.imshow('edge_and_corner', edge_image(gray))

    elif cur_mode == ord('c'):
        cv2.imshow('edge_and_corner', corner_image(frame, gray))
        # cv2.imshow('edge_or_corner', good_track(frame, gray))

    elif cur_mode == ord('s'):
        cv2.imshow('edge_and_corner', sift_image(frame, gray))

    else:
        cv2.imshow('edge_and_corner', frame)


cap.release()
cv2.destroyAllWindows()
