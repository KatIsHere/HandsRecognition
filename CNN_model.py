from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPool2D, MaxPooling2D
from keras.utils import np_utils
from image_read import read_data, classLen
from keras.callbacks import ModelCheckpoint
import numpy as np
"""Implements a convolutional neural network and trains it on a set of given images using keras"""

# training/testing set
(X_train, Y_train), (X_test, Y_test) = read_data()

# normalizing data 
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

batch_size, img_rows, img_colums = 32, 50, 50
nb_clases = classLen()

# converting images into a 2d array
X_train = X_train.reshape(X_train.shape[0], img_rows, img_colums, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_colums, 1)
input_shape = (img_rows, img_colums, 1)     # in 1 colour
Y_train = np_utils.to_categorical(Y_train, nb_clases)
Y_test = np_utils.to_categorical(Y_test, nb_clases)

# formalizing model
model = Sequential()

# 2 layers of neuron network:

# first layer
model.add(Convolution2D(32, (5, 5), border_mode = "same", input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), border_mode = "same"))
# second layer
model.add(Convolution2D(64, 5, 5, border_mode = "same", input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), border_mode = "same"))

# tensor reforming 
model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))     # reg
model.add(Dense(nb_clases))
model.add(Activation("softmax"))

# optimizing
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
# learning
model.fit(X_train, Y_train, callbacks = [ModelCheckpoint("model.hdf5", monitor = "val_acc",      \
            save_best_only = True, save_weights_only = False, mode = "auto")], nb_epoch = 10,    \
            verbose = 1, validation_data = (X_test, Y_test))
# accuracy
score = model.evaluate(X_test, Y_test, verbose = 0)
print("Test score :", score[0])
print("Test accuracy :", score[1], "\n")

# saving model to .json file
model_json = model.to_json()
with open("CNN_model.json", 'w') as json_file:
    json_file.write(model_json)
# saving weights to .hdf5 file
model.save_weights("CNN_model.h5")
print("Model saved\n")