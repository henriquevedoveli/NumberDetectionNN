
import numpy as np
import cv2 
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.models import load_model

###
path = 'myData'
imgDim = (64,64,3)

batch = 50
epoch = 20
steps = 130

###

imgs = []
classNo = []
myList = os.listdir(path)
noOfClasses = len(myList)

print('Total de Classes detectadas:',  len(myList))
print('Importando Dados...')

for i in range(0,noOfClasses):
    myPicList = os.listdir(path+'/'+str(i))
    for j in myPicList:
        curImg = cv2.imread(path+'/'+str(i)+'/'+j)
        curImg = cv2.resize(curImg, (imgDim[0], imgDim[1]))
        imgs.append(curImg)
        classNo.append(i)

    print(i, end = ' ')


imgs = np.array(imgs)
classNo = np.array(classNo)


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(imgs, classNo, test_size = 0.20) 
# GETING VALIDATION DATA
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20) 


noOfSamples =[]
for i in range(0, noOfClasses):
    noOfSamples.append(len(np.where(y_train==i)[0]))


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255

    return img

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_val = np.array(list(map(preProcessing, X_val)))


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)


dataGen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            shear_range=0.1,
                            rotation_range=30)

dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)

# based on LeNet Model
def model():
    filters = 60
    sizeFilter1 = (5,5)
    sizeFilter2 = (3,3)
    sizePool = (2,2)
    node = 5000

    model = Sequential()
    model.add((Conv2D(filters, sizeFilter1, input_shape = (imgDim[0],imgDim[1],1),activation='relu')))
    model.add((Conv2D(filters, sizeFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizePool))
    model.add((Conv2D(filters//2, sizeFilter2, activation='relu')))
    model.add((Conv2D(filters//2, sizeFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizePool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(5000,activation='relu'))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

    return model

model = model()
print(model.summary())

history = model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size=batch),
                                 steps_per_epoch=steps,
                                 epochs=epoch,
                                 validation_data=(X_val,y_val),
                                 shuffle=1)

#### PLOT THE RESULTS  
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

#### EVALUATE USING TEST IMAGES
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

#### SAVE THE TRAINED MODEL 
model.save('model20.h5')


