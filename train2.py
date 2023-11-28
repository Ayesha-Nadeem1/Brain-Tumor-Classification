# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:55:04 2023

@author: Ayesha Nadeem
"""

import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

data_path = 'dataset/'
dataset = []
label = []
image_size = 64

no_tumor_images = os.listdir(data_path + 'Training/no_tumor/')
glioma_tumor_images = os.listdir(data_path + 'Training/glioma_tumor/')
meningioma_tumor_images = os.listdir(data_path + 'Training/meningioma_tumor/')
pituitary_tumor_images = os.listdir(data_path + 'Training/pituitary_tumor/')

for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(data_path + 'Training/no_tumor/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((image_size, image_size))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(glioma_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(data_path + 'Training/glioma_tumor/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((image_size, image_size))
        dataset.append(np.array(image))
        label.append(1)

for i, image_name in enumerate(meningioma_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(data_path + 'Training/meningioma_tumor/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((image_size, image_size))
        dataset.append(np.array(image))
        label.append(2)

for i, image_name in enumerate(pituitary_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(data_path + 'Training/pituitary_tumor/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((image_size, image_size))
        dataset.append(np.array(image))
        label.append(3)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(x_train)

# Model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(image_size, image_size, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4))  # Assuming you have 4 classes
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Learning Rate Scheduler
def lr_scheduler(epoch):
    return 0.001 * np.exp(-epoch / 10)

scheduler = LearningRateScheduler(lr_scheduler)

# Train the model with data augmentation
model.fit(datagen.flow(x_train, y_train, batch_size=16), epochs=20, validation_data=(x_test, y_test), callbacks=[scheduler])

model.save('BrainTumorImproved.h5')
