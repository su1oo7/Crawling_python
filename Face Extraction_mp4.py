#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os, sys
import cv2

capture = cv2.VideoCapture('test3.mp4')
face_cascade = cv2.CascadeClassifier(r'C:\Users\USER\Anaconda3\envs\selftest\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
overlap_naming=1
while(True):
    ret, frame = capture.read()
    #read the camera image
    #카메라에서 이미지 얻기
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #(Blue, Green, Red 계열의 이미지를 gray이미지로 변환. BGR2GRAY)    
    faces = face_cascade.detectMultiScale(grayframe, 1.1, 4 )
    #gray로 변환된 이미지를 cascade를 이용하여 detect

#   face_list = cascade.detectMultiScale(image_gs,scaleFactor = 1.1, minNeighbors = 4, minSize = (70,70))
    

    if(int(capture.get(1)) % 20 == 0):
        for face in faces:
            x,y,w,h = face
            resizing_img = cv2.resize(frame[y:y+h, x:x+w], dsize=(64,64), interpolation = cv2.INTER_AREA)
            cv2.imwrite('test_cap3/'+str(overlap_naming)+'.jpg', resizing_img)
            overlap_naming += 1
            print(overlap_naming)


# In[ ]:




