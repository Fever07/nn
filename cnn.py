import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPool2D, Dropout, Flatten, Dense
from prepare_data import prepare_batch_data
from tensorflow.keras.models import load_model

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

model = keras.Sequential()
model.add(Conv2D(9, (3, 3), padding='same', activation='relu', input_shape=[256, 256, 3]))
model.add(Conv2D(9, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(9, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(9, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(196, activation='relu'))
model.add(Dense(2, activation='softmax'))

batch_size = 32
epochs = 2
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

for i in range(10):
    trainx, testx, trainy, testy = prepare_batch_data()
    model.fit(trainx, trainy, batch_size=batch_size, epochs=epochs, verbose=1,
                         validation_data=(testx, testy))

model.save('model_cnn.h5')
print('Model has been saved')



