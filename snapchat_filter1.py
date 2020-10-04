import numpy as np
import cv2

cap=cv2.VideoCapture(0)

face_cascade        = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_alt.xml')
eyes_cascade        = cv2.CascadeClassifier("cascades\\frontalEyes35x16.xml")
nose_cascade        = cv2.CascadeClassifier("cascades\\Nose18x15.xml")
glasses             = cv2.imread("images\\glasses.png",-1)
mustache            = cv2.imread("images\\mustache.png",-1)


while(True):
    ret,frame=cap.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    for (x, y, w, h) in faces:
        roi_gray    = gray[y:y+h, x:x+h] 
        roi_color   = frame[y:y+h, x:x+h]
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            glasses2 =cv2.resize(glasses,(ew,eh), interpolation=cv2.INTER_AREA) 
            gw, gh, gc = glasses2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    if glasses2[i, j][3]!= 0: 
                        roi_color[ey + i, ex + j] = glasses2[i, j]

        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (nx, ny, nw, nh) in nose:
            #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
            roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
            mustache2 = cv2.resize(mustache,(nw,nh), interpolation=cv2.INTER_AREA) 
            mw, mh, mc = mustache2.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    if mustache2[i, j][3] != 0:
                        roi_color[ny + i+int(nh/2), nx+ j] = mustache2[i, j]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow('frame',frame)
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()