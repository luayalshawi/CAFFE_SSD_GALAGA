import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while True:
    dict = {}
    # img = ImageGrab.grab(bbox=(0,0,1680,1050),backend='scrot') #bbox specifies specific region (bbox= x,y,width,height)
    # frame = np.array(img)
    flag, frame = cap.read()
    cv2.imshow('video', frame)

    cv2.waitKey(1)
