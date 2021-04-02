'''Ini kontur yang diajarin Murtaza
cari berapa sisinya
shape nya ada dimana dan lain sebagainya '''


import numpy as np
import cv2
import imutils

def getContour(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnts in contours: #buat misahin lalu di loopin masing masing kontur
        area = cv2.contourArea(cnts)

        #biar areanya bisa menyesuakian ketika ada benda lain
        areaMin = cv2.getTrackbarPos('Area', 'Tracking')

        if area > areaMin: #area nya harus lebih dari 1000, biar bisa drawcontour, kalo gabisa yaudah
            cv2.drawContours(imgContour, cnts, -1, (255,0,255), 7)
            peri = cv2.arcLength(cnts, True)
            approx = cv2.approxPolyDP(cnts, 0.02 * peri, True)
            print(len(approx))  #kalo di print keluar berapa corner point
            
            '''0.02 itu teh factor, yang memperjelas corner'''

            #membuat rectangle dari approx
            x, y, w, h = cv2.boundingRect(approx)
            print(x,y,w,h)
            cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0, 255, 0), 5)

            #PutText biar keren
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255,0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x+w+20, y+45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255,0), 2)
            




#Fungsi buat nge stack images
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





cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

#create TRackbar
def nothing(x):
    pass

cv2.namedWindow('Tracking')
cv2.createTrackbar('th1', 'Tracking', 38, 255, nothing)
cv2.createTrackbar('th2', 'Tracking', 122, 255, nothing)
cv2.createTrackbar('Area', 'Tracking', 5000, 30000, nothing)


while True :
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    imgContour = frame.copy()
    img_blur = cv2.GaussianBlur(frame, (7,7), 0)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)



    #Trackbar pos & Canny
    th1 = cv2.getTrackbarPos('th1', 'Tracking')
    th2 = cv2.getTrackbarPos('th2', 'Tracking')
    canny = cv2.Canny(img_gray, th1, th2)

    #dilate
    kernel = np.ones([6,6]) #buat mempertebal garis putih
    img_dil = cv2.dilate(canny, kernel)

    #fungsi getContour
    getContour(img_dil, imgContour)
    



    imgStack = stackImages(0.8, ([frame, img_gray, canny, imgContour]))
    
    
    
    cv2.imshow('frame', imgStack)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()