import handtrackingmodule as htm
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import os
import sys

#Reading the headers
folderpath = "Header"
path = os.path.join(sys.path[0],folderpath)
list = os.listdir(path)
headers = []
for i in list:
    filepath = os.path.join(path,i)
    headers.append(cv.imread(filepath))

#Reading the video
cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

#Initiation
header = headers[0]
drawcolor = (204,153,255)
xp,yp = 0 ,0
brushThickness = 10
imgcanvas = np.zeros((720,1280,3),np.uint8)
detector = htm.hand_detector(detect_confidence = 0.85)


while True: 
    # Reading the image 
    succes, frame = cap.read()
    frame = cv.flip(frame,1) 
    
    # Finding the landmarks
    frame = detector.findHands(frame,draw = False)
    lmlist = detector.findposition(frame,draw = False)

    if(len(lmlist)):

        #Tip of fingers
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
        
        # Check which fingers are up
        fingers = detector.fingersUp()
    
        # Selection mode -two fingers are up
        if  not fingers[1] and not fingers[2]:
            xp,yp = 0 ,0
            cv.rectangle(frame,(x1,y1-25),(x2,y2+25),(0,0,255),cv.FILLED)
            print("selection mode")
            if y1 < 100:
                if 209 < x1 < 400:
                    header = headers[0]
                    drawcolor = (204,153,255)
                elif 477 < x1 < 585:
                    header = headers[1]
                    drawcolor = (242,78,66)
                elif 685 < x1 < 796:
                    header = headers[2]
                    drawcolor = (77,226,31)
                elif 916 < x1 < 1080:
                    header = headers[3]
                    drawcolor = (0,0,0)
        # Drawing mode - one finger is up
        if not fingers[1] and not (fingers[2] == False):
            cv.circle(frame,(x1,y1),15,drawcolor,cv.FILLED)
            print("Drawing mode") 

            #Drawing
            if (xp ==0 and yp ==0):
                xp, yp = x1, y1
            cv.line(frame,(xp,yp),(x1,y1),drawcolor, brushThickness)
            cv.line(imgcanvas,(xp,yp),(x1,y1),drawcolor, brushThickness)
            xp,yp = x1,y1
    # Setting the header image
    frame[0:100,0:1080] = header
    
    imgcanvas_bw = cv.cvtColor(imgcanvas,cv.COLOR_BGR2GRAY)
    _ ,imgINV = cv.threshold(imgcanvas_bw,50,255,cv.THRESH_BINARY_INV)
    imgINV = cv.cvtColor(imgINV,cv.COLOR_GRAY2BGR)
    frame = cv.bitwise_and(frame,imgINV)
    frame = cv.bitwise_or(frame,imgcanvas)
    
    cv.rectangle(frame,(0,150),(1080,100),(180,223,6),cv.FILLED)
    cv.putText(frame,"Use right hand to draw",(370,132),cv.FONT_HERSHEY_SIMPLEX,fontScale = 1, color = (255,255,255),thickness = 4)
    cv.imshow("image",frame)
    key = cv.waitKey(1)
    if key == 27:
        break
cap.release()
cv.destroyAllWindows()