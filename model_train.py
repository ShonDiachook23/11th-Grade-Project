import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

train_dir = "./changed_train"
len(os.listdir(train_dir))

IMG_SIZE = 60
CATEGORIES = ["cat", "dog"]

def load_train_data(image_directory="./changed_train"):
    X_train, X_test, Y_train, Y_test = [], [], [], []

    global CATEGORIES
    global ToClass
    for cls in CATEGORIES:
        count = 0
        currX = []
        currY = []
        ToClass = {"cat": 0, "dog": 1}
        for file_name in os.listdir(os.path.join(image_directory, cls)):
            count += 1
            img = Image.open(os.path.join(image_directory, cls, file_name)).convert('L')  # Convert to grayscale
            img = img.resize((IMG_SIZE, IMG_SIZE))

            img_array = np.array(img).reshape((IMG_SIZE, IMG_SIZE, 1)) / 255.0
            currX.append(img_array)
            currY.append(ToClass[cls])

            if count % 1000 == 0:
                print(f"Loaded {count} images from {cls} class")
            if count == 2000:
                break

        # split the data set equally
        currX_train, currX_test, currY_train, currY_test = train_test_split(currX, currY, test_size=0.2, random_state=42)
        for element in currX_train:
            X_train.append(element)

        for element in currX_test:
            X_test.append(element)

        for element in currY_train:
            Y_train.append(element)

        for element in currY_test:
            Y_test.append(element)

    X_train, Y_train, X_test, Y_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
    print("\n\nData loaded successfully\n\n")
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    print("\n\n")

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = load_train_data()

# create the model
input_shape = (IMG_SIZE, IMG_SIZE, 1)
model = keras.Sequential()
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.30))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, epochs=70, batch_size=256)

# save the model
model.save("model.h5")