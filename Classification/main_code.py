import cv2
import numpy as np
from keras.models import model_from_json
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QPalette, QBrush
from PyQt5.QtCore import pyqtSlot, QSize
import sys

class App(QWidget):
    global basariOrani
    basariOrani = ""

    def __init__(self):
        super().__init__()
        self.title = 'Sayısal İşaret İşleme Proje'
        self.left = 450
        self.top = 100
        self.width = 1027
        self.height = 806
        self.imagePath=""
        oImage = QImage("cs.jpg")
        sImage = oImage.scaled(QSize(1200, 806))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(sImage))
        self.setPalette(palette)
        # self.setStyleSheet("background:rgba(71,71,71,255);")
        self.initUI()

    def initUI(self):
        global button2
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        button = QPushButton('Resim Yükle', self)
        button.move(281, 690)
        button.setFixedWidth(161)
        button.setFixedHeight(61)
        button.setObjectName("yukle")

        button2 = QPushButton('Gönder', self)
        button2.move(70, 690)
        button2.setFixedWidth(161)
        button2.setFixedHeight(61)
        button2.setObjectName("gonder")

        button3 = QPushButton('Resim Çek', self)
        button3.move(485, 690)
        button3.setFixedWidth(161)
        button3.setFixedHeight(61)
        button3.setObjectName("resimCek")

        button.setStyleSheet("#yukle{font: 75 10pt \"Microsoft YaHei UI\";\n""font-weight: bold;\n"
                             "color: rgb(255, 255, 255);\n""background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.5, y2:0.5,x3:1, y3:1, "
                             "stop:0 rgb(255, 255, 255), stop:1 rgb(0, 0, 0));\n""border-style: solid;\n""border-radius:21px;\n""}"
                             "#yukle:hover{background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.5, y2:0.5,x3:1, y3:1, "
                             "stop:0 rgba(255, 255, 255,100), stop:1 rgba(0, 0, 0,100));}")

        button2.setStyleSheet("#gonder{font: 75 10pt \"Microsoft YaHei UI\";\n""font-weight: bold;\n"
                              "color: rgb(255, 255, 255);\n""background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.5, y2:0.5,x3:1, y3:1, "
                              "stop:0 rgb(255, 255, 255), stop:1 rgb(0, 0, 0));\n""border-style: solid;\n""border-radius:21px;\n""}"
                              "#gonder:hover{background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.5, y2:0.5,x3:1, y3:1, "
                              "stop:0 rgba(255, 255, 255,100), stop:1 rgba(0, 0, 0,100));}")
        button3.setStyleSheet("#resimCek{font: 75 10pt \"Microsoft YaHei UI\";\n""font-weight: bold;\n"
                              "color: rgb(255, 255, 255);\n""background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.5, y2:0.5,x3:1, y3:1, "
                              "stop:0 rgb(255, 255, 255), stop:1 rgb(0, 0, 0));\n""border-style: solid;\n""border-radius:21px;\n""}"
                              "#resimCek:hover{background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.5, y2:0.5,x3:1, y3:1, "
                              "stop:0 rgba(255, 255, 255,100), stop:1 rgba(0, 0, 0,100));}")
        button2.setEnabled(False)
        button.clicked.connect(self.on_click)
        button2.clicked.connect(self.gonder)

        self.label = QLabel(self)
        self.label.move(50, 50)
        self.label.setFixedWidth(621)
        self.label.setFixedHeight(611)
        self.label.setStyleSheet("border-radius:20px; background-color:white;")
        button3.clicked.connect(self.resim_cek)
        self.label2 = QLabel(self)
        self.label2.move(710, 50)
        self.label2.setFixedWidth(256)
        self.label2.setFixedHeight(311)
        self.label2.setStyleSheet("border-radius:20px; background-color:white;")

        self.label3 = QLabel(self)
        self.label3.move(710, 399)
        self.label3.setFixedWidth(256)
        self.label3.setFixedHeight(130)
        self.label3.setStyleSheet(
            "font-weight:bolder ;border-radius:20px; background-color:white;font-size:25px;color:black;text-align:center")
        self.label3.setText("  Basari Orani : " + str(basariOrani))

        self.label4 = QLabel(self)
        self.label4.move(710, 490)
        self.label4.setFixedWidth(256)
        self.label4.setFixedHeight(130)
        self.label4.setStyleSheet("border-radius:20px; background-color:white;font-size:25px;color:black;")
        self.label4.setText("  Nesne Türü   : " + str(basariOrani))
        self.show()

    def resim_cek(self):
    
        cap = cv2.VideoCapture(0)
        cap.set(1,640)
        cap.set(1,480)
        cap.set(10,50)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        while True:
            success, img = cap.read()
            # videos is gray
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,4)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 1)
        
            cv2.imshow("faces",img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                img = img[y:y+h+10, x:x+w+10]
                print(type(img),img.ndim)
                img = cv2.resize(img,(128,128))
                img = np.asarray(img)
                cv2.imwrite("/home/yunus/Sayisal_proje/test_set/human/resim.jpg",img)
                print("resim kaydedildi")
                break
            
        self.imagePath = "/home/yunus/Sayisal_proje/test_set/human/resim.jpg"
        pixmap = QPixmap(self.imagePath)
        pixmap = pixmap.scaled(621, 611)
        pixmap2 = pixmap.scaled(256, 311)
        self.label.setPixmap(pixmap)
        self.label.adjustSize()
        self.label2.setPixmap(pixmap2)
        self.label2.adjustSize()
        button2.setEnabled(True)
        
        cap.release()
        cv2.destroyAllWindows()
    def on_click(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.*)")

        self.imagePath = image[0]
        pixmap = QPixmap(self.imagePath)
        pixmap = pixmap.scaled(621, 611)
        pixmap2 = pixmap.scaled(256, 311)
        self.label.setPixmap(pixmap)
        self.label.adjustSize()
        self.label2.setPixmap(pixmap2)
        self.label2.adjustSize()
        button2.setEnabled(True)



    def gonder(self):
        
        frame=self.imagePath
        frame = cv2.imread(frame) 
       # print(frame)
        def preProcess(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = img / 255

            return img

        model = model_from_json(open("modelveri.json", "r").read())
        model.load_weights("weight_veri.h5")

        img = np.asarray(frame)
      #  print(img)
        img = cv2.resize(img, (64, 64))
        img = preProcess(img)

        img = img.reshape(-1, 64, 64, 1)

        classindex = np.argmax(model.predict(img), axis=-1)
        print(classindex)

        predic = model.predict(img) 
        proVal = np.amax(predic)    
        print(classindex, proVal)   

        if proVal > 0.5 and classindex == 0:
            basari = "Ağaç" 
        elif proVal > 0.5 and classindex == 1:
            basari = "Kedi" 
        elif proVal > 0.5 and classindex == 2:
            basari = "Köpek" 
        elif proVal > 0.5 and classindex == 3:
            basari = "İnsan" 
        else: 
            basari = "nesne turu belirlenemedi"

        print(self.imagePath)
        self.label3.setText("  Basari Orani :% " + str(int(proVal*100)))
        self.label4.setText("  Nesne Türü   : " + str(basari))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
