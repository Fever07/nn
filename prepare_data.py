import os
from PIL import Image
import numpy
from sklearn.model_selection import train_test_split as tts
from tensorflow import keras

image_path = 'E:\PROJECTS\images\Histology_CAMELYON16_300K_Tiles\Pink_Tiles\\'
norm, tum = 'Normal', 'Tumor'
curr_batch = 0
batch = 500

npath = os.path.join(image_path, norm)
npaths = os.listdir(npath)
print('Normal images count: %s ...' % len(npaths))
tpath = os.path.join(image_path, tum)
tpaths = os.listdir(tpath)
print('Tumor images count: %s ...' % len(tpaths))

abs_npaths = [os.path.join(npath, img_npath) for img_npath in npaths]
abs_tpaths = [os.path.join(tpath, img_npath) for img_npath in tpaths]
print('Finished joining...')


def get_batch_data():

    global curr_batch

    data = []
    labels = []

    # normal
    print('Start batching normal images from %s to %s...' % (curr_batch, curr_batch + batch))
    for i in range(curr_batch, curr_batch + batch):
        img = Image.open(abs_npaths[i])
        arr = numpy.array(img)[:, :, :3]
        data.append(arr)
        labels.append(0)

    print('Normal images batched...')

    # tumor
    print('Start batching tumor images from %s to %s...' % (curr_batch, curr_batch + batch))
    for i in range(curr_batch, curr_batch + batch):
        img = Image.open(abs_tpaths[i])
        arr = numpy.array(img)[:, :, :3]
        data.append(arr)
        labels.append(1)

    print('Tumor images batched...')
    curr_batch += batch
    return numpy.array(data), numpy.array(labels)


def prepare_batch_data():
    data = get_batch_data()
    trainx, testx, trainy, testy = tts(data[0], data[1], test_size=0.2, stratify=data[1], random_state=42)
    trainx = trainx / 255.0
    testx = testx / 255.0
    trainy = keras.utils.to_categorical(trainy)
    testy = keras.utils.to_categorical(testy)
    return trainx, testx, trainy, testy

