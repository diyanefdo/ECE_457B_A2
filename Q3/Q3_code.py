from matplotlib.cbook import flatten
import numpy as np
from tables import Description
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.8, random_state=42)
X_validation = test_images
y_validation = test_labels

print(X_train.shape)

# Normalization of the image data
X_train = X_train / 255.0
X_validation = X_validation / 255.0

weight_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
bias_init = tf.keras.initializers.Zeros()

def net_1():
    cnn = Sequential()
    # Image data needs to be flattened to run through fully connected layer
    cnn.add(Flatten())
    cnn.add(Dense(512, activation='sigmoid', kernel_initializer=weight_init, bias_initializer=bias_init))
    cnn.add(Dense(512, activation='sigmoid', kernel_initializer=weight_init, bias_initializer=bias_init))
    cnn.add(Dense(10, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = cnn.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_validation, y_validation))
    
    print("training accuracy: ", history.history["accuracy"][-1])
    print("validation accuracy: ", history.history["val_accuracy"][-1])
    plt.plot(history.history['loss'], "r-")
    plt.plot(history.history['val_loss'], "b-")
    plt.show()


def net_2():
    cnn = Sequential()
    cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    cnn.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    cnn.add(Flatten())
    cnn.add(Dense(512, activation='sigmoid'))
    cnn.add(Dense(512, activation='sigmoid'))
    cnn.add(Dense(10, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = cnn.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_validation, y_validation))

    print("training accuracy: ", history.history["accuracy"][-1])
    print("validation accuracy: ", history.history["val_accuracy"][-1])
    plt.plot(history.history['loss'], "r-")
    plt.plot(history.history['val_loss'], "b-")
    plt.show()

def net_3():
    cnn = Sequential()
    cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    # max pooling layer by default is 2 x 2
    cnn.add(MaxPool2D())
    cnn.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    cnn.add(MaxPool2D())
    cnn.add(Flatten())
    cnn.add(Dense(512, activation='sigmoid'))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(512, activation='sigmoid'))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(10, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = cnn.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_validation, y_validation))

    print("training accuracy: ", history.history["accuracy"][-1])
    print("validation accuracy: ", history.history["val_accuracy"][-1])
    plt.plot(history.history['loss'], "r-")
    plt.plot(history.history['val_loss'], "b-")
    plt.show()

net_1()
net_2()
net_3()