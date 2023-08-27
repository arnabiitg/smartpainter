import cv2 as cv
import numpy as np
import mediapipe as mp
import time


class hand_detector():
    def __init__(self,mode=False,hands=1,detect_confidence=0.5,tracking_confidence=0.5):
        self.mode = mode
        self.maxhands = hands
        self.detect = detect_confidence
        self.tracking = tracking_confidence
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(mode, hands,1,detect_confidence,tracking_confidence)
        self.draw = mp.solutions.drawing_utils
        self.tipids = [4,8,12,16,20]
        self.lmlist =[]
    
    
    def findHands(self,frame,draw = True):
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_RGB)
        if draw:
            if self.results.multi_hand_landmarks:
                for handlms in  self.results.multi_hand_landmarks:
                    self.draw.draw_landmarks(frame, handlms,self.mphands.HAND_CONNECTIONS)
        return frame   
    
    def findposition(self,frame,handno = 0,draw =True):
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            for handlms in  self.results.multi_hand_landmarks: 
                for id,lm in enumerate(handlms.landmark):
                    h,w,c= frame.shape
                    cx,cy = int((lm.x)*w),int((lm.y)*h)
                    ##print(id,f"x:{cx}\ny:{cy}")
                    self.lmlist.append([id,cx,cy])
                    if draw:
                        if(id == handno):
                            cv.circle(frame,(cx,cy),5,(255,255,0),thickness = 5)
        return self.lmlist
    
    def fingersUp(self):
        fingers = []
        if self.lmlist[self.tipids[0]][1] > self.lmlist[self.tipids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if self.lmlist[self.tipids[id]][2] > self.lmlist[self.tipids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    cap = cv.VideoCapture(0)
    ptime = 0
    ctime = 0
    detection = hand_detector()
    while True:
        success, frame = cap.read()
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        frame = detection.findHands(frame)
        list = detection.findposition(frame,8)
        cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
        cv.imshow('video',frame)
        key = cv.waitKey(1)
        if key == 27:
            break
    print(detection.lmlist)
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()