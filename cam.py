# Tracking white golf balls with my webcam

import cv2
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)
cv2.namedWindow("image")

if cap.isOpened():
    ret, img = cap.read()
else:
    ret = False

while ret:

    # cv2.imshow("image", img)
    ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0,0,213], dtype=np.uint8)
    upper_white = np.array([172,111,255], dtype=np.uint8)

    # threshold the HSV image to get only white color
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # residual pixels matching the color mask...
    res = cv2.bitwise_and(img,img, mask= mask)
    
    # find contours for box tracking
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    x,y,w,h = cv2.boundingRect(cont_sorted[0])
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
    
    # three windows, og image with tracking, residual pixels, grayscale
    cv2.imshow('image',img) # og
    cv2.imshow('mask',mask) # grayscale
    cv2.imshow('res',res)   # residual

    key = cv2.waitKey(20)
    if key == 27: # close on ESC
        break

cv2.destroyAllWindows()
cap.release()
