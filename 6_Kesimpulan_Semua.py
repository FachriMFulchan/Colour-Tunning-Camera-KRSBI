'''
Oke tahapannya gini
1. Masking dengan HSV
2. Ngebentuk Kontur dan Centroid
3. Dapetin Koordinat
4. Tunning warna dan Besar dari Kontur
5. Stack Images kalo butuh (minimal 3 frame)
Tapi masih ada yang kurang
dilanjut di versi berikutnya
'''


import numpy as np
import cv2
import imutils

#Fungsi buat StackImages
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver






#Fungsi For Detect Ball
def CariBola(frame, ukuran, LowerColor, UpperColor):
    if frame is None:
        pass
    else:
        #Preprocessing
        frame = imutils.resize(frame, width= 320)
        blurred = cv2.GaussianBlur(frame, (11,11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        #masking
        mask = cv2.inRange(hsv, LowerColor, UpperColor)

        #Correction
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        
        center = None

        if len(cnts) <= 0:
            pass

        else:
            for c in cnts:
                area = cv2.contourArea(c)
                areaMin = cv2.getTrackbarPos('Area', 'Tracking')
                
                #Reduce Noise dengan Area
                if area > areaMin:
                    
                    #Enclosing Circle
                    ((x,y), radius) = cv2.minEnclosingCircle(c)

                    #Moments
                    M = cv2.moments(c)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, (cx,cy), 7, (255,0,0), -1)
                    
                    cv2.putText(frame, "centre", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
                    cv2.putText(frame, "dx: {}, dy: {}".format(int(x), int(y)),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1)
                    print ("areas is ...", area)
                    print ("centroid at is...", cx, cy)
            
        imgStack = stackImages(0.8, ([hsv, mask, frame]))  
        
        return imgStack


#Setting Camera dan Resolusi
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
kernel = np.ones((6,6), np.uint8)


#Membuat Trackbar
def nothing(x):
    pass

cv2.namedWindow('Tracking')
cv2.resizeWindow('Tracking', width = 500, height = 300)
cv2.createTrackbar('Upper H', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('Upper S', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('Upper V', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('Lower H', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower S', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Area', 'Tracking', 5000, 30000, nothing)

#Set Initial Trackbar (Sunnah)
cv2.setTrackbarPos('Upper H', 'Tracking', 17)
cv2.setTrackbarPos('Upper S', 'Tracking', 255)
cv2.setTrackbarPos('Upper V', 'Tracking', 255)
cv2.setTrackbarPos('Lower H', 'Tracking', 0)
cv2.setTrackbarPos('Lower S', 'Tracking', 143)
cv2.setTrackbarPos('Lower V', 'Tracking', 0)



while True:
    ret, frame = cap.read()

    #Get Trackbar Pos
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
    cv2.imshow('frame', frame)


    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


'''Masih ada kekurangan
karena harus ngatur Luas Area

dan kalo cnts di for in terus jadi c
ngatur Luas Area nya harus bener bener 
biar ga ada 2 kontur yang terbentuk



kalo pake c = max(cnts, key=cv2.contourArea)
dibandingin mana yang lebih gede cntst atau contourArea

jadi cuma salah satu kontur aja yang ke tampil
Setting area ga terlalu pengaruh
tapi boleh juga ditambahin
'''