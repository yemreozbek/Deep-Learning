import glob
import os
import numpy as np
from keras.layers import Dense, Dropout, Flatten #katman, seyreltme, duzlestirme
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D #evrisim agi, pixsel ekleme
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

imgs = glob.glob("./img/*.png")

width = 125
height = 50

X = []
Y = []

for img in imgs:
    
    filename = os.path.basename(img)
    label = filename.split("_")[0]
    im = np.array(Image.open(img).convert("L").resize((width,height)))
    im = im / 255
    X.append(im)
    Y.append(label)
    
X = np.array(X)
X = X.reshape(X.shape[0],width,height,1) #kac tane resim oldugu,genislik,yukselkik,channel degeri

#sns.countplot(Y) #neden kac tane tiklanmis

def onehot_labels(values):
    label_encoder = LabelEncoder()
    #up down right gibi text ifadeleri 1 0 2 yaptik
    integer_encoded = label_encoder.fit_transform(values)
    one_hot_encoded = OneHotEncoder(sparse = False)
    #(379,) lu olan vektoru (379,1) yaptik
    integer_encoded =   integer_encoded.reshape(len(integer_encoded),1)
    one_hot_encoded =   one_hot_encoded.fit_transform(integer_encoded)
    return one_hot_encoded

Y = onehot_labels(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.27, random_state = 2)

#CNN MODEL
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3),
          activation = "relu",
          input_shape = (width, height, 1)))

model.add(Conv2D(64, kernel_size = (3,3),
          activation = "relu"))

model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(3, activation = "softmax"))

#EGITILMIS MODELIN YUKLENMESI
#if os.path.exists("model ismi(./trex_weights.h5)"):
#    model.load_weights("model ismi")
#    print("yuklendi")
    




model.compile(loss = "categorical_crossentropy",
              optimizer = "Adam", metrics = ["accuracy"])

model.fit(x_train, y_train, epochs = 35, batch_size = 64)

score_train = model.evaluate(x_train, y_train)
print("Egitim dogrulugu: %",score_train[1]*100)


score_test = model.evaluate(x_test, y_test)
print("Test dogrulugu: %",score_train[1]*100)

#EGITILEN MODELIN KAYDEDILMESI
open("model.json","w").write(model.to_json())
model.save_weights("trex_weight.h5")

