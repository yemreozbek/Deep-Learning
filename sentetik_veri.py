import os 
import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import display


def show(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) -1) // n_cols + 1
    
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis = -1)
    plt.figure(figsize=(n_cols, n_rows))
    
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap= "binary")
        plt.axis("off")

#%% Verilerin yuklenmesi ve on islemlerin gerceklestirilmesi
#fashion veri kumesinin indirilmesi
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

#veri kumesinden 10x10 piksel buyuklugunde 25 tane ornek ekrana yazdirilmasi
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap = plt.cm.binary)
plt.show()

#%% egitim verileri ve gruplarinin olusturulmasi
batch_size = 32
#bu veri kumesi, bir arabellegi buffer_size elemanlari ile doldurur,
#ardindan secilen elemanlari yeni elemanlarla degistirerek rastgele ornekler.

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
#bu veri kumesinin ardisik ogelerini toplu olarak birlestirir. 


#%% URETICI KATMANINDAKI EVRISIMLI SINIR AGI
num_features = 100 #oznitelik sayisi

#giris degerini verdigimiz features sayisina gore baslatiyoruz
#Conv2DIraspose versiyonunu kullaniyoruz.

generator = keras.models.Sequential([
        keras.layers.Dense(7 * 7 * 128, input_shape = [num_features]),
        keras.layers.Reshape([7,7,128]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(64,(5,5), (2,2), padding = "same", activation="selu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(1, (5,5), (2,2), padding= "same", activation = "tanh"),
        ])

#uretileck olan veri aslinda suanlik sadece gurultu
noise = tf.random.normal(shape = [1, num_features])
generated_images = generator(noise, training=False)
show(generated_images,1)

#%%DCGAN OLUSTURULMASI
discriminator = keras.models.Sequential([
        keras.layers.Conv2D(64, (5,5), (2,2), padding = "same", input_shape = [28,28,1]),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128,(5,5),(2,2),padding = "same"),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(1,activation = "sigmoid")
        
    ])
#uretilen gorsel icin ayirt edici %50 nin altinda bir deger uretti ilk adim icin
decion = discriminator(generated_images)
print(decion)

discriminator.compile(loss = "binary_crossentropy", optimizer = "rmsprop")
discriminator.trainable = False
gan = keras.models.Sequential([generator, discriminator])
gan.compile(loss = "binary_crossentropy", optimizer = "rmsprop")

#%% Egitim islemlerini gorsellestirilmesi
seed = tf.random.normal(shape = [batch_size, 100])

def generate_and_save_images(model, epoch, test_input):
    #egitim false secenegine ayarlandi
    #boylece tum katmalar cikarim modunda calisir
    
    predictions = model(test_input, training = False)
    
    fig = plt.figure(figsize=(10,10))
    
    for i in range(25):
        plt.subplot(5,5, i+1)
        plt.imshow(predictions[i,:,:,0] * 127.5 + 127.5, cmap = "binary")
        plt.axis("off")
    
    plt.savefig("image_at_epoch_{:04d}.png")
    plt.show()

def train_dcgan(gan, dataset, batch_size, num_features, epochs=5):
    generator, discriminator = gan.layers
    for epoch in tqdm(range(epochs)):
        print("Epoch {}/{}".format(epoch +1,epochs))
        for X_batch in dataset:
            noise = tf.random.normal(shape = [batch_size, num_features])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis = 0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            noise = tf.random.normal(shape = [batch_size, num_features])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
            
            #uretilen goruntuleri ekrana yazdirip dosyaya kaydedelim
        
        display.clear_output(wait = True)
        generate_and_save_images(generator, epoch +1, seed)
    
    display.clear_output(wait = True)
    generate_and_save_images(generator, epochs, seed)
    

#%% DCGAN'in Egitilmesi
#egitim icin yeniden boyutlandirilmasi
    
x_train_dcgan = x_train.reshape(-1,28,28,1) * 2. -1.

#Batch size boyutunun ve suffle ozelliklerinn belirlenmesi
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(x_train_dcgan)
dataset = dataset.shuffle(1000) #data karistirma
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

#egitim gan modeli, tanimlanan dataset icin belirlenen batch_size boyutu ve oznitelik sayisi ile 
#10 epoch olarak gerceklesecek, siz de epoch sayisini degistirerek modelin gelisimini gozlemle

train_dcgan(gan, dataset, batch_size,num_features,epochs=10)
    
    
#%% Sonuclarin gif olarak olusturulmasi
import imageio
import glob

anim_file = "dcgan.gif"

with imageio.get_writer(anim_file, mode = "I") as writer:
    filenames = glob.glob("image*.png")
    filenames = sorted(filenames)
    last = -1
    
    for i, filename in enumerate(filenames):
        frame = 2*(i)
        if round(frame) > round(last):
            last = frame
        else:
            continue
    
        image = imageio.imread(filename)
        writer.append_data(image)
    
    image = imageio.imread(filename)
    writer.append_data(image)
    
    from IPython.display import Image
    Image(open(anim_file,"rb").read())
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
