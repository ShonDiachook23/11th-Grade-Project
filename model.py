from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from DL3 import *
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns

image_size = (64, 64) 
classNames = ["cat", "dog"]

def load_train_data(image_directory="./changed_train"):
    X_train, X_test, Y_train, Y_test = [], [], [], []

    global classNames
    global ToClass
    for cls in classNames:
        count = 0
        currX = []
        currY = []
        ToClass = {"cat": 0, "dog": 1}
        for file_name in os.listdir(os.path.join(image_directory, cls)):
            count += 1
            img = Image.open(os.path.join(image_directory, cls, file_name))
            img = img.resize(image_size)    

            img_array = np.array(img).reshape(image_size[0]*image_size[1]*3,) / 255.0 - 0.5
            currX.append(img_array)
            currY.append(ToClass[cls])

            print(f"Loaded {cls}/{file_name}")
            if (count == 2000):
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


# Load the data
X_train, X_test, Y_train, Y_test = load_train_data()
Y_test = Y_test.T
Y_train = Y_train.T
X_train = X_train.T
X_test = X_test.T

np.random.seed(19)


# Define the model
model = DLModel("Dog or Cat Classifier")
model.add(DLLayer("Input Layer",  32, (64*64*3,), activation="sigmoid", learning_rate=0.2, random_scale=0.01))
model.add(DLLayer("Output Layer",  1, (32,), activation="sigmoid", learning_rate=0.2, random_scale=0.01))

model.compile("cross_entropy")
costs = model.train(X_train, Y_train, 1000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()
Y_prediction_train = model.predict(X_train)
Y_prediction_test = model.predict(X_test)

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

model.save_weights(f"saved_weights")