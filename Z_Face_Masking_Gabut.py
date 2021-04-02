import numpy as np
import cv2
import imutils


def CariBola (frame, ukuran, LowerColor, UpperColor, LowerColor2, UpperColor2):
    if frame is None:
        pass
    else:
        frame = imutils.resize(frame, width = 320)
        blurred = cv2.GaussianBlur(frame, (11,11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, LowerColor, UpperColor)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('mask', mask)

        mask2 = cv2.inRange(hsv, LowerColor2, UpperColor2)
        mask2 = cv2.erode(mask2, None, iterations=2)
        mask2 = cv2.dilate(mask2, None, iterations=2)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('mask2', mask2)

        
        res = cv2.bitwise_and(frame, frame, mask=mask)
        res2 = cv2.bitwise_and(frame, frame, mask=mask2)

        auh = cv2.bitwise_or(res, res2)

        return auh


cap = cv2.VideoCapture(0)
kernel = np.ones((6,6), np.uint8)

#create Trackbar
def nothing(x):
    pass

cv2.namedWindow('Tracking')
cv2.resizeWindow('Tracking', 350, 250)
cv2.createTrackbar('Lower H', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower S', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Upper H', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('Upper S', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Tracking', 255, 255, nothing)

#Set initial Trackbar
cv2.setTrackbarPos('Lower H', 'Tracking', 7)
cv2.setTrackbarPos('Lower S', 'Tracking', 66)
cv2.setTrackbarPos('Lower V', 'Tracking', 140)
cv2.setTrackbarPos('Upper H', 'Tracking', 24)
cv2.setTrackbarPos('Upper S', 'Tracking', 255)
cv2.setTrackbarPos('Upper V', 'Tracking', 255)

cv2.namedWindow('Tracked')
cv2.resizeWindow('Tracked', 350, 250)
cv2.createTrackbar('Lower H', 'Tracked', 0, 255, nothing)
cv2.createTrackbar('Lower S', 'Tracked', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'Tracked', 0, 255, nothing)
cv2.createTrackbar('Upper H', 'Tracked', 255, 255, nothing)
cv2.createTrackbar('Upper S', 'Tracked', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Tracked', 255, 255, nothing)

#Set initial Trackbar
cv2.setTrackbarPos('Lower H', 'Tracked', 7)
cv2.setTrackbarPos('Lower S', 'Tracked', 66)
cv2.setTrackbarPos('Lower V', 'Tracked', 140)
cv2.setTrackbarPos('Upper H', 'Tracked', 24)
cv2.setTrackbarPos('Upper S', 'Tracked', 255)
cv2.setTrackbarPos('Upper V', 'Tracked', 255)


while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h1 = cv2.getTrackbarPos('Lower H', 'Tracking')
    l_s1 = cv2.getTrackbarPos('Lower S', 'Tracking')
    l_v1 = cv2.getTrackbarPos('Lower V', 'Tracking')
    u_h1 = cv2.getTrackbarPos('Upper H', 'Tracking')
    u_s1 = cv2.getTrackbarPos('Upper S', 'Tracking')
    u_v1 = cv2.getTrackbarPos('Upper V', 'Tracking')


    l_h2 = cv2.getTrackbarPos('Lower H', 'Tracked')
    l_s2 = cv2.getTrackbarPos('Lower S', 'Tracked')
    l_v2 = cv2.getTrackbarPos('Lower V', 'Tracked')
    u_h2 = cv2.getTrackbarPos('Upper H', 'Tracked')
    u_s2 = cv2.getTrackbarPos('Upper S', 'Tracked')
    u_v2 = cv2.getTrackbarPos('Upper V', 'Tracked')

    #dibuat aray buat boundaries
    l_b1 = np.array([l_h1, l_s1, l_v1])
    u_b1 = np.array([u_h1, u_s1, u_v1])

    l_b2 = np.array([l_h2, l_s2, l_v2])
    u_b2 = np.array([u_h2, u_s2, u_v2])


    frame = CariBola(frame, 10, l_b1, u_b1, l_b2, u_b2)

    # #buat mask
    # mask = cv2.inRange(hsv, l_b, u_b)

    # #masking
    # res = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow('frame', frame)
    # cv2.imshow('hsv', hsv)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()