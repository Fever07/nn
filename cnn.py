import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPool2D, Dropout, Flatten, Dense
from prepare_data import prepare_batch_data
from tensorflow.keras.models import load_model


model = keras.Sequential()
model.add(Conv2D(9, (3, 3), padding='same', activation='relu', input_shape=[256, 256, 3]))
model.add(Conv2D(9, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(9, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(9, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(9, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(196, activation='relu'))
model.add(Dense(2, activation='softmax'))

batch_size = 100
epochs = 50
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='binary_crossentropy', metrics=['accuracy'])

trainx, testx, trainy, testy = prepare_batch_data()
model.fit(trainx, trainy, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(testx, testy))

model.save('model_cnn.h5')
print('Model has been saved')



