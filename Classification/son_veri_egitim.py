#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:40:22 2021

@author: yunus
"""

#%% kutuphanelerin ice aktarilmasi
import numpy as np
import cv2
import keras
import os #verileri ice aktarmak icin
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.backend import backend as K
import pickle #modeli kaydetmek icin
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam

path = "tree_train"

mylist = os.listdir(path)
output = len(mylist)

images = []
classno = []

"""

 
"""

#14 farkli agac turu var
for i in mylist:
    myImageList = os.listdir(path + "//"+str(i))
    for j in myImageList:
        img = cv2.imread(path+ "//"+str(i)+"//"+j) # artik dosyalarin icinde bulunan resimlere erisebildik
        img = cv2.resize(img,(64,64)) #resim boyutu
        images.append(img)
        classno.append(i)
        
print(len(images))
print(len(classno))

lb =LabelBinarizer()
classno = lb.fit_transform(classno)
#classno = to_categorical(classno, num_classes=3)

images = np.array(images)
classno = np.array(classno)

#%% Veri Ayirma
x_train, x_test, y_train, y_test = train_test_split(images,classno, test_size = 0.33,stratify=classno, random_state = 42)
#x_train, x_validation, y_train, y_validation = train_test_split(x_train,y_train, test_size = 0.33, random_state = 42)

print(images.shape)
print(x_train.shape)
print(x_test.shape)
#print(x_validation.shape)

#%% vis
"""
fig, axes = plt.subplots(3,1,figsize=(5,7))
fig.subplots_adjust(hspace = 0.5)
sns.countplot(y_train, ax = axes[0])
axes[0].set_title("y train")

sns.countplot(y_test, ax = axes[1])
axes[1].set_title("y test")

sns.countplot(y_validation, ax = axes[2])
axes[2].set_title("y validation")
"""
#%% preProcess
#bu kismi egitim asamasina dahil etme !!!!!

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    
    return img


img = preProcess(x_train[100])
img = cv2.resize(img,(300,300))
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


x_train = np.array(list(map(preProcess, x_train))) #map fonksiyonu 2 parametre alir ve 2.yi 1.isleme sokar
x_test = np.array(list(map(preProcess, x_test))) 
#x_validation = np.array(list(map(preProcess, x_validation)))

x_train = x_train.reshape(-1,64,64,1) #burdaki -1 xtrain in boyutunun uzunlugunun otomatik ayarlanmasinin saglar
x_test = x_test.reshape(-1,64,64,1)
#x_validation = x_validation.reshape(-1,64,64,1)

#data generate
dataGen = ImageDataGenerator(width_shift_range = 0.08,
                             height_shift_range= 0.08,
                             zoom_range= 0.05, # 0.05
                             rotation_range= 16,
                             fill_mode="nearest",
                             horizontal_flip=True)

dataGen.fit(x_train)

#%% CNN MODEL
model = Sequential()
model.add(Conv2D(input_shape = (64,64,1), filters = 16, kernel_size= (2,2), activation="relu",padding="valid")) #valid -> same, filtre icin 4 -> 8
model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = "valid"))

model.add(Conv2D(input_shape = (31,31,1), filters = 32, kernel_size = (2,2), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = "valid"))
model.add(Conv2D(input_shape = (15,15,1), filters = 32, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(Conv2D(input_shape = (15,15,1), filters = 32, kernel_size= (2,2), activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = "valid"))

model.add(Conv2D(input_shape = (7,7,1), filters = 8, kernel_size = (1,1), activation = "relu", padding = "valid"))
model.add(MaxPooling2D(pool_size=(1,1), strides = (1,1), padding = "valid"))

model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(units=64, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(units=output, activation = "softmax")) 

model.compile(loss = "categorical_crossentropy", optimizer = Adam(), metrics = ["accuracy"])

batch_size = 10 #
model.summary()
hist = model.fit_generator(dataGen.flow(x_train,y_train, batch_size = batch_size),
                                        steps_per_epoch=len(x_train) // batch_size, shuffle = 1, # shuffle veriyi karistirir.
                                        validation_data = (x_train, y_train),
                                        validation_steps = len(x_test) // batch_size,
                                        epochs=50)

open("modelveri2.json","w").write(model.to_json())
model.save_weights("weight_veri2.h5")



# model score
model.summary()
score = model.evaluate(x_test,y_test, verbose=1)
print("test lose: ",score[0])
print("test accuracy:",score[1])

#%% modelin gecmisteki loss degerlerini gosterir
print(model.history.history.keys())
model.history.history["loss"]


