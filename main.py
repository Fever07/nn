import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# prepare data
cancer = load_breast_cancer()
data = cancer.data
value = cancer.target
trainx, testx, trainy, testy = train_test_split(data, value, test_size=0.2, stratify=value, random_state=42)
ss = StandardScaler()
ss.fit(trainx)
trainx, testx = ss.transform(trainx), ss.transform(testx)

# build model
model = keras.Sequential()
model.add(keras.layers.Dense(200, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(keras.layers.Dense(1, activation='relu'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',
              metrics=['accuracy'])

model.fit(trainx, trainy, epochs=1000, validation_data=(testx, testy))
import numpy
with open('model.npz', 'wb') as file:
    ar = []
    for layer in model.layers:
        wts = layer.get_weights()
        ar.append(numpy.array(len(wts)))
        for arr in wts:
            ar.append(arr)
    numpy.savez(file, *ar)
