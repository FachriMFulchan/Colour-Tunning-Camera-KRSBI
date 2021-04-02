'''Masking Bola lalu menambahkan kontur acak'''


import cv2
import numpy as np
import imutils


#generate function
def CariBola(frame, ukuran, LowerColor, UpperColor):
    if frame is None:
        pass
    else:
        frame = imutils.resize(frame, width=320)
        blurred = cv2.GaussianBlur(frame, (11,11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, LowerColor, UpperColor)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('mask', mask)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        

        return frame




cap = cv2.VideoCapture(0)
kernel = np.ones((6,6), np.uint8)


def nothing(x):
    pass

#Create Trackbar
cv2.namedWindow('Tracking', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Upper H', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('Upper S', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('Lower H', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower S', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'Tracking', 0, 255, nothing)

#Set initial Trackbar
cv2.setTrackbarPos('Upper H', 'Tracking', 30)
cv2.setTrackbarPos('Upper S', 'Tracking', 255)
cv2.setTrackbarPos('Upper V', 'Tracking', 255)
cv2.setTrackbarPos('Lower H', 'Tracking', 0)
cv2.setTrackbarPos('Lower S', 'Tracking', 28)
cv2.setTrackbarPos('Lower V', 'Tracking', 166)

while True:
    ret, frame = cap.read()


    u_h = cv2.getTrackbarPos('Upper H', 'Tracking')
    u_s = cv2.getTrackbarPos('Upper S', 'Tracking')
    u_v = cv2.getTrackbarPos('Upper V', 'Tracking')

    l_h = cv2.getTrackbarPos('Lower H', 'Tracking')
    l_s = cv2.getTrackbarPos('Lower S', 'Tracking')
    l_v = cv2.getTrackbarPos('Lower V', 'Tracking')

    u_b = np.array([u_h, u_s, u_v])
    l_b = np.array([l_h, l_s, l_v])

    frame = CariBola(frame, 10, l_b, u_b)
    cv2.imshow('frame', frame)



    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


#Kenapa Blurred dulu baru HSV??
#Karena kalo HSV dulu kacau warnanya

#Erode terus di dilate, sebenernya ga terlalu pengaruh banyak
#kalo iterationnya cuma 2
#tapi lumayan nge reduce noise waktu bolanya di goyang goyang
#kalo iteration kebanyakan nanti banyak yang kekikis
#maka itu ditambahin lagi Morphlogy Ex