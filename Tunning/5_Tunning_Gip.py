'''Ini Tunning Copas dari Kang Gip'''


import numpy as np
import cv2
import imutils
import argparse
from collections import deque

def CariBola(frame, ukuran, LowerColor, UpperColor):
    if frame is None:
        pass

    else:
        '''HSV Detection'''
        frame = imutils.resize(frame, width=320)      #height nya menyesuaikan dengan width
        blurred = cv2.GaussianBlur(frame, (11,11), 0)  #Remove Noise
        
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LowerColor, UpperColor)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)  #di erode terus dilate sama aja, tapi lebih stabil hasilnya
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('Perspective_Mask'+ str(ukuran), mask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        center = None

        #only proceec if at least one countour was found
        if len(cnts) <= 0:
            pass
        
        #find the largest contour in the mask, then use it
        # to compute the minimum enclosing circle and centroid
        else:
            c = max(cnts, key=cv2.contourArea)
            print(cv2.contourArea)
            ((x,y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            #only proceed if the radius meets a minimums size
            if radius > ukuran:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)
                            
            # loop over the set of tracked points
            cv2.putText(frame, "dx: {}, dy: {}".format(int(x-300), int(y-240)),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1)
        return frame


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
        help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
        help="max buffer size")
args = vars(ap.parse_args())


pts = deque(maxlen=args["buffer"])



cap = cv2.VideoCapture(0)
kernel = np.ones((6,6), np.uint8)

def nothing(x):
    pass

#Trackbar
cv2.namedWindow('Tracking')
cv2.createTrackbar('Upper H', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Upper S', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Upper V', 'Tracking', 0, 255, nothing)

cv2.createTrackbar('Lower H', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower S', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'Tracking', 0, 255, nothing)

#Set Initial Trackbar
cv2.setTrackbarPos('Lower H', 'Tracking', 2)
cv2.setTrackbarPos('Lower S', 'Tracking', 178)
cv2.setTrackbarPos('Lower V', 'Tracking', 58)
cv2.setTrackbarPos('Upper H', 'Tracking', 41)
cv2.setTrackbarPos('Upper S', 'Tracking', 255)
cv2.setTrackbarPos('Upper V', 'Tracking', 255)


while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    u_h = cv2.getTrackbarPos('Upper H', 'Tracking')
    u_s = cv2.getTrackbarPos('Upper S', 'Tracking')
    u_v = cv2.getTrackbarPos('Upper V', 'Tracking')
    
    l_h = cv2.getTrackbarPos('Lower H', 'Tracking')
    l_s = cv2.getTrackbarPos('Lower S', 'Tracking')
    l_v = cv2.getTrackbarPos('Lower V', 'Tracking')


    #Dibuat array
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    frame = CariBola(frame, 10, l_b, u_b)

    cv2.imshow('Perspective', frame)




    ''' Comment dulu ya maap'''
    # #Terus di InRange hsv nya buat create mask
    # mask = cv2.inRange(hsv, l_b, u_b)

    # #Terus frame nya di mask
    # res = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow('frame', frame)
    # cv2.imshow('hsv', hsv)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)



    k = cv2.waitKey(1) 
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#Bagus cuma banyak Noise
#Karena gadikasih filter dan lain lain

#hsv --> buat bikin mask
#frame --> terus di masking oleh inRange HSV
