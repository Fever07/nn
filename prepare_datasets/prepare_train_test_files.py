import os
from PIL import Image
import numpy
from sklearn.model_selection import train_test_split as tts
from tensorflow import keras
import sys

absp = '/media/data10T_1/datasets/Voynov/Histology_CAMELYON16_300K_Tiles'
norm, tum = 'Normal', 'Tumor'
blue, pink = 'Blue_Tiles', 'Pink_Tiles'

blue_tumor = os.path.join(absp, blue, tum)
pink_tumor = os.path.join(absp, pink, tum)
btumor = os.listdir(blue_tumor)
ptumor = os.listdir(pink_tumor)

btumor = numpy.array([os.path.join(blue_tumor, p) for p in btumor])
ptumor = numpy.array([os.path.join(pink_tumor, p) for p in ptumor])
sl = numpy.arange(0, 1000)
indices = numpy.empty(0, dtype=numpy.int32)
for i in range(25):
    indices = numpy.append(indices, sl + 6000 * i)
btumor = btumor[indices]
ptumor = ptumor[indices]
tumor = numpy.append(btumor, ptumor)
labels = numpy.ones([50000], dtype=numpy.int32)

file = open('pink.txt', 'r')
lines = file.readlines()
normal = lines[:25000]
normal = [p.split(' ')[0] for p in normal]
file.close()
file = open('blue.txt', 'r')
lines = file.readlines()
normal2 = lines[:25000]
normal2 = [p.split(' ')[0] for p in normal2]
normal = normal + normal2
labels = numpy.append(labels, numpy.zeros([50000], dtype=numpy.int32))
paths = numpy.append(tumor, normal)

trainp, testp, trainy, testy = tts(paths, labels, test_size=0.3, stratify=labels, random_state=42)
file = open('train.txt', 'w')
for p, y in zip(trainp, trainy):
    file.write(p + ' ' + str(y) + '\n')
file.close()
file = open('test.txt', 'w')
for p, y in zip(testp, testy):
    file.write(p + ' ' + str(y) + '\n')
file.close()
