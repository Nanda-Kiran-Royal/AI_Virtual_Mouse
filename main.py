import time

import cv2 as cv
import cvzone
from cvzone import HandTrackingModule
import numpy as np
from pynput.mouse import Button,Controller
import pyautogui



wCam,hCam = 1920,1080
pTime = 0
smoothening = 6
plocX,plocY = 0,0
clocX,clocY = 0,0
detector = HandTrackingModule.HandDetector(detectionCon=0.8,maxHands=1 )
mouse = Controller()
size = pyautogui.size()
wScr = size[0]
hScr = size[1]
frameR = 100
cap = cv.VideoCapture(0)

cap.set(2,wCam)
cap.set(4,hCam)


while True:
    # 1. Find the hand landmarks
    success,img = cap.read()
    img = cv.flip(img,1)


    #2 Get the tip of middle finger and index
    hands,img = detector.findHands(img,flipType=False)
    if hands:
        hand1 = hands[0]
        lmList1 = hand1['lmList']
        bbox1 = hand1['bbox']


        x1,y1  = lmList1[8][0],lmList1[8][1]
        x2,y2 = lmList1[12][0],lmList1[12][1]


        # print(x1,y1,x2,y2)
        # 3 Check which fingers are up
        fingers = detector.fingersUp(hand1)
        # print(fingers)
        # 4 Only index finger moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            cv.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # print(wScr,hScr)


            # Convert Coordinates
            #6 Smoothen the values
            clocX = plocX+ (x3-plocX)/smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(clocX,clocY)
            cv.circle(img,(x1,y1),15,(255,0,255),cv.FILLED)
            plocX,plocY = clocX,clocY
        # 7 Move Mouse

        # 8 Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            length,lineInfo,img = detector.findDistance((lmList1[8][0],lmList1[8][1]),(lmList1[12][0],lmList1[12][1]),img)
            # print(length)
            # 10 Click mouse if distance is short
            if length<55:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                cv.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv.FILLED)
                pyautogui.click(x3,y3)


    #Find Distance between fingers
    ## Set Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,str(int(fps)),(20,50),cv.FONT_HERSHEY_SIMPLEX,3,(255,0,0),3)



    cv.imshow('Image',img)
    cv.waitKey(1)