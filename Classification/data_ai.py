#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 19:30:26 2021

@author: yunus
"""

import cv2
import numpy as np
from keras.models import model_from_json

frame = cv2.imread(r"/home/yunus/Sayisal_proje/test_set/human/resim.jpg", cv2.IMREAD_COLOR)

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    
    return img


model = model_from_json(open("modelveri.json","r").read())
model.load_weights("weight_veri.h5")

img = np.asarray(frame)
img = cv2.resize(img,(64,64))
img = preProcess(img)

img = img.reshape(-1,64,64,1)

classindex = np.argmax(model.predict(img), axis = -1)
print(classindex)
    
predic = model.predict(img)
proVal = np.amax(predic)
print(classindex, proVal)

if proVal > 0.5 and classindex==0:
    cv2.putText(frame, "tree"+"  "+str(proVal),(20,20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))
elif proVal> 0.5 and classindex==1:
    cv2.putText(frame, "cat"+"  "+str(proVal),(20,20),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
elif proVal> 0.5 and classindex==2:
    cv2.putText(frame, "dog"+"  "+str(proVal),(20,20),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
    
elif proVal> 0.5 and classindex==3:
   cv2.putText(frame, "human"+"  "+str(proVal),(20,20),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))

else:
    cv2.putText(frame, "Nesne Turu Belirlenemedi"+"  "+str(proVal),(20,20),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))


cv2.imshow("goruntu",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()