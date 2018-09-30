import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPool2D, Dropout, Flatten, Dense


data = keras.datasets.mnist.load_data()
(train_img, train_val), (test_img, test_val) = data
train_img = train_img / 255.0
test_img = test_img / 255.0

model = keras.Sequential()
model.add(Conv2D(1, (3, 3), padding='same', activation='relu', input_shape=[28, 28, 1]))
model.add(Conv2D(1, (3, 3), padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(196, activation='relu'))
model.add(Dense(10, activation='softmax'))

train_val = keras.utils.to_categorical(train_val)
test_val = keras.utils.to_categorical(test_val)

batch_size = 256
epochs = 2
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_img.reshape([60000, 28, 28, 1]), train_val, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(test_img.reshape([10000, 28, 28, 1]), test_val))
import numpy
with open('model_cnn.npz', 'wb') as file:
    ar = []
    for layer in model.layers:
        wts = layer.get_weights()
        ar.append(numpy.array(len(wts)))
        for arr in wts:
            ar.append(arr)
    numpy.savez(file, *ar)

