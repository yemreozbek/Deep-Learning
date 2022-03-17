#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:59:36 2021

@author: yunus
"""

import cv2
from keras.models import model_from_json
import numpy as np

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    
    return img

cap = cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,480)

model = model_from_json(open("modelveri.json","r").read())
model.load_weights("weight_veri.h5")

while True:
    success, frame = cap.read()
    
    
    img = np.asarray(frame)
    img = cv2.resize(img,(64,64))
    img = preProcess(img)
    
    img = img.reshape(1,64,64,1)
    
    #predict
    
    classindex = np.argmax(model.predict(img), axis = -1)
    print(classindex)
    
    predic = model.predict(img)
    proVal = np.amax(predic)
    print(classindex, proVal)
    if proVal > 0.5 and classindex==0:
        cv2.putText(frame, "human"+"  "+str(proVal),(50,50),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0))
    elif proVal> 0.5 and classindex==1:
        cv2.putText(frame, "cat"+"  "+str(proVal),(50,50),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,0))
    elif proVal> 0.5 and classindex==2:
        cv2.putText(frame, "tree"+"  "+str(proVal),(50,50),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,0))
    
    elif proVal> 0.5 and classindex==3:
       cv2.putText(frame, "dog"+"  "+str(proVal),(50,50),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,0))
    
    
    cv2.imshow("goruntu",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
