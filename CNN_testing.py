import numpy as np
import cv2
from keras.models import load_model

########################
width = 640
height = 480
threshold = 0.60

########################

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

model = load_model('model20.h5')

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255

    return img

while True:
    sucess, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(64,64))
    img = preProcessing(img)
    # cv2.imshow("Processsed Image",img)
    img = img.reshape(1,64,64,1)

    # PREDICT
    classIdx = int(model.predict_classes(img))
    pred = model.predict(img)

    probVal = np.amax(pred)

    if probVal > threshold:
        cv2.putText(imgOriginal,str(classIdx)+ '  ' + str(probVal),(50,50),cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)


    cv2.imshow('ORIGINAL IMAGE',imgOriginal)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break