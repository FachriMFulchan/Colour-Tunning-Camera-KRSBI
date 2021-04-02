'''Masking bola menurut warna'''


import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 550)
cap.set(4, 550)


def nothing(x):
    pass

#Trackbar
cv2.namedWindow('Tracking')
cv2.createTrackbar('Lower H', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower S', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Upper H', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Upper S', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Upper V', 'Tracking', 0, 255, nothing)

#Set Initial Trackbar
cv2.setTrackbarPos('Lower H', 'Tracking', 2)
cv2.setTrackbarPos('Lower S', 'Tracking', 26)
cv2.setTrackbarPos('Lower V', 'Tracking', 153)
cv2.setTrackbarPos('Upper H', 'Tracking', 16)
cv2.setTrackbarPos('Upper S', 'Tracking', 218)
cv2.setTrackbarPos('Upper V', 'Tracking', 255)


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos('Lower H', 'Tracking')
    l_s = cv2.getTrackbarPos('Lower S', 'Tracking')
    l_v = cv2.getTrackbarPos('Lower V', 'Tracking')

    u_h = cv2.getTrackbarPos('Upper H', 'Tracking')
    u_s = cv2.getTrackbarPos('Upper S', 'Tracking')
    u_v = cv2.getTrackbarPos('Upper V', 'Tracking')
    
    #Dibuat array
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    





    #Terus di InRange hsv nya buat create mask
    mask = cv2.inRange(hsv, l_b, u_b)

    #Terus frame nya di mask
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('hsv', hsv)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#Bagus cuma banyak Noise
#Karena gadikasih filter dan lain lain

#hsv --> buat bikin mask
#frame --> terus di masking oleh inRange HSV
