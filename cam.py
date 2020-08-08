# Track white objects with webcam

import cv2
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)
cv2.namedWindow("image")
# img = cv2.resize(vid, (320, 220))
# face_locations = []

if cap.isOpened():
    ret, img = cap.read()
else:
    ret = False

while ret:

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_frame = img[:, :, ::-1]

    # cv2.imshow("image", img)
    ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # face_locations = face_recognition.face_locations(rgb_frame)
    lower_white = np.array([0,0,213], dtype=np.uint8)
    upper_white = np.array([172,111,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    x,y,w,h = cv2.boundingRect(cont_sorted[0])
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)

    ####### Facial rec stuff #######
    # for top, right, bottom, left in face_locations:
    #
    #     # Draw a box around the face
    #     cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

    cv2.imshow('image',img)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

# cv2.destroyWindow("image")
cv2.destroyAllWindows()
cap.release()