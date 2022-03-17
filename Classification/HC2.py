import cv2
import numpy as np

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


cap.release()
cv2.destroyAllWindows()