import cv2 as cv
import numpy as np
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
draw = mp.solutions.drawing_utils
ptime = 0
ctime = 0
while True:
    success, frame = cap.read()
    if success:
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_RGB)
        ctime = time.time()
        fps = 1/(ctime-ptime)
        cv.putText(frame,str(int(fps)),(10,70),3,cv.FONT_HERSHEY_PLAIN,color= (0,0,255),thickness = 2)
        if results.multi_hand_landmarks:
            for handlms in  results.multi_hand_landmarks:
                for id,lm in enumerate(handlms.landmark):
                    
                    h,w,c= frame.shape
                    cx,cy = int((lm.x)*w),int((lm.y)*h)
                    print(id,f"x:{cx}\ny:{cy}")
                    if(id == 8):
                        cv.circle(frame,(cx,cy),5,(255,255,0),thickness = 5)
                draw.draw_landmarks(frame, handlms,mphands.HAND_CONNECTIONS)
        cv.imshow('video',frame)
    ptime = ctime    
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()