import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense




data_path = 'dataset/'
dataset=[]
label=[]
image_size = 64

no_tumor_images = os.listdir(data_path+'Training/no_tumor/')
glioma_tumor_images = os.listdir(data_path+'Training/glioma_tumor/')
meningioma_tumor_images = os.listdir(data_path+'Training/meningioma_tumor/')
pituitary_tumor_images = os.listdir(data_path+'Training/pituitary_tumor/')

# print(glioma_tumor_images)
# print(cv2.__version__)
for i, image_name in enumerate (no_tumor_images): 
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(data_path+'Training/no_tumor/'+image_name) 
        image=Image.fromarray(image, 'RGB') 
        image=image.resize((image_size,image_size))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate (glioma_tumor_images): 
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(data_path+'Training/glioma_tumor/'+image_name) 
        image=Image.fromarray(image, 'RGB') 
        image=image.resize((image_size,image_size))
        dataset.append(np.array(image))
        label.append(1)

for i, image_name in enumerate (meningioma_tumor_images): 
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(data_path+'Training/meningioma_tumor/'+image_name) 
        image=Image.fromarray(image, 'RGB') 
        image=image.resize((image_size,image_size))
        dataset.append(np.array(image))
        label.append(2)

for i, image_name in enumerate (pituitary_tumor_images): 
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(data_path+'Training/pituitary_tumor/'+image_name) 
        image=Image.fromarray(image, 'RGB') 
        image=image.resize((image_size,image_size))
        dataset.append(np.array(image))
        label.append(3)

# print(len(dataset))
# print(len(label))

dataset= np.array(dataset)
label= np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset,label,test_size=0.2,random_state=0)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# print(np.__version__)
# print(tf.__version__)
x_train = normalize(x_train,axis=1)
x_test = normalize(x_test,axis=1)

#model 

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(image_size, image_size, 3))) 
model.add(Activation( 'relu'))
model.add(MaxPooling2D (pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform')) 
model.add(Activation('relu'))
model.add(MaxPooling2D (pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform')) 
model.add(Activation('relu'))
model.add(MaxPooling2D (pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout (0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=12,validation_data=(x_test,y_test),shuffle=False)

model.save('BrainTumor10Epochs.h5')