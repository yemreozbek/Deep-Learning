#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:22:29 2020

@author: root
"""

from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

mon = {"top":360, "left":505, "width":250, "height":100} 
sct = mss()

width = 125
height = 50

#model yukle
model = model_from_json(open("model.json","r").read())
model.load_weights("trex_weight.h5")

#down=0, right = 1, up =2
labels = ["Down","Right","Up"]
frametare_time = time.time()
counter = 0
i =0
delay = 0.4 #bekleme suresi

key_down_presseed = False

#down =0, up =2, right = 1
while True:
    
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im2 = np.array(im.convert("L").resize((width,height)))
    im2 = im2 / 255
    
    X = np.array([im2])
    X = X.reshape(X.shape[0],width,height,1)
    r = model.predict(X)
    
    result = np.argmax(r)
    
    if result == 0: #down = 0
        keyboard.press(keyboard.KEY_DOWN)
        key_down_presseed = True
    elif result == 2: #up = 2
        if key_down_presseed:
            keyboard.release(keyboard.KEY_DOWN)
        
        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)
        
        keyboard.release(keyboard.KEY_UP)
        if i<1200:
            time.sleep(0.30)
        elif 1200 < i and i < 4000:
            time.sleep(0.20)
        else:
            time.sleep(0.17)
        
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
    
    counter += 1
    
    if(time.time() - frametare_time > 1):
        print("zaman",time.time() - frametare_time)
        counter = 0
        frametare_time = time.time()
        if(i<=1200):
            delay -= 0.003
        else:
            delay -= 0.005
        if(delay < 0):
            delay = 0
        
        
        print("-------------------")
        print("down: {} \nRight: {} \nUp: {}\n".format(r[0][0],r[0][1],r[0][2]))
        i += 1



















