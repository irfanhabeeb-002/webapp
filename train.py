import os
from PIL import Image
import numpy as np
import os
 

X = []
y = []

base_path='./dataset/'
for child in os.listdir(base_path):
    sub_path = os.path.join(base_path, child)
    if os.path.isdir(sub_path):
        for data_file in os.listdir(sub_path):
            X_i = Image.open(os.path.join(sub_path, data_file))
            X_i = np.array(X_i.resize((120,120))) / 255.0
            X.append(X_i)
            y.append(child)
print(np.shape(X))
print(y)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), 
                                                    test_size=0.2, random_state=42)
#X_train=X_train.reshape([-1,120,120,1])
#X_test=X_test.reshape([-1,120,120,1])


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

cnnModel = models.Sequential()

cnnModel.add(layers.Conv2D(8, (3,3), activation="relu",
                           input_shape=(120, 120, 3)))
cnnModel.add(layers.MaxPooling2D((2,2)))
cnnModel.add(layers.Conv2D(16, (3,3), activation="relu"))
cnnModel.add(layers.MaxPooling2D((2,2)))

cnnModel.add(layers.Flatten())

# Reduce number of neurons in dense layers
cnnModel.add(layers.Dense(32, activation="relu"))
cnnModel.add(layers.Dense(16, activation="relu"))

cnnModel.add(layers.Dense(1, activation="sigmoid"))

cnnModel.summary()


cnnModel.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])


cnnModel.fit(X_train, y_train, epochs=20, batch_size=32)

testLoss, testAccuracy = cnnModel.evaluate(X_test, y_test)

print(testAccuracy)

cnnModel.save("./mymodel.keras")