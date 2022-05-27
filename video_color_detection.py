import cv2
import numpy as np

color_name = str(input('Please enter color (r/b/g): '))
if color_name == 'r' or color_name == 'b' or color_name == 'g':
    print('Color  found')
else:
    print('Color not found')
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera issue')
    exit()

while True:
    rat, frame = cap.read()
    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])

    lower_blue = np.array([94, 80, 20])
    upper_blue = np.array([126, 255, 255])

    lower_green = np.array([40, 40, 40])
    upper_green =  np.array([90, 255,255]) 

    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask3 = cv2.inRange(hsv, lower_green, upper_green)

    if color_name == 'r':
        res = cv2.bitwise_and(frame, frame, mask = mask)
        mask0 = mask
    elif color_name == 'b':
        res = cv2.bitwise_and(frame, frame, mask = mask2)
        mask0 = mask2
    elif color_name == 'g':
        res = cv2.bitwise_and(frame, frame, mask = mask3)
        mask0 = mask2

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask0)
    cv2.imshow('res', res)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()