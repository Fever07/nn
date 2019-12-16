import os
import sys
from datetime import datetime
import numpy as np
from numpy.random import seed
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow import set_random_seed
from tensorflow.keras.models import Model

from core.utils_nn import parse_file, load_gray_image, load_color_image, get_model_path, get_folder_name
from core.generator import ConfigurationGenerator, Generator, AttacksGenerator
from core.pgd_attack import PGD_attack as pgd
import pickle
from tqdm import tqdm

from tensorflow.keras.applications.inception_v3 import InceptionV3
from scipy.stats.stats import pearsonr

from skimage import transform

input_shape = [256, 256, 1]
n_classes = 2
batch_size = 20
images_are_colored = input_shape[-1] == 3
train_file = 'train.txt'
test_file = 'test.txt'
model_name = 'inceptionv3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

absp = '../__aorta_razv'
model_path = get_model_path(absp, model_name=model_name)
model = load_model(model_path)
train_generator = Generator(os.path.join(absp, train_file), batch_size=batch_size, colored=images_are_colored, num_classes=n_classes)
test_generator = Generator(os.path.join(absp, test_file), batch_size=batch_size, colored=images_are_colored, num_classes=n_classes)
write_corrs = True
write_feats = True

feature_model = Model(inputs=model.input, outputs=model.layers[-3].output)
if write_corrs:
    features = np.zeros(shape=[train_generator.total, 6, 6, 2048], dtype=np.float32)
    ys = np.zeros(shape=[train_generator.total, n_classes], dtype=np.float32)
    for i in tqdm(range(len(train_generator))):
        X, y = train_generator[i]
        # print('Corrs {}/{}'.format(i, len(train_generator)))
        batch_feats = feature_model.predict(X)
        features[i * batch_size: (i + 1) * batch_size] = batch_feats
        ys[i * batch_size: (i + 1) * batch_size] = y

    averages = np.mean(features, axis=(1, 2))
    corrs = []
    for i in range(averages.shape[1]):
        corrs.append(pearsonr(averages[:, i], ys[:, 1])[0])

    with open(os.path.join(absp, model_name, 'corrs.pkl'), 'wb') as file:
        pickle.dump(corrs, file)

if write_feats:
    split = 10
    step = int(len(test_generator) / split)
    for s in tqdm(range(split)):
        features = np.zeros(shape=[int(test_generator.total / split), 6, 6, 2048], dtype=np.float32)
        for j in tqdm(range(step)):
            i = s * step + j
            X, y = test_generator[i]
            batch_feats = feature_model.predict(X)
            features[j * batch_size: (j + 1) * batch_size] = batch_feats

        if not os.path.isdir(os.path.join(absp, model_name, 'feats')):
            os.mkdir(os.path.join(absp, model_name, 'feats'))

        with open(os.path.join(absp, model_name, 'feats', 'test_feats_{}.pkl'.format(s)), 'wb') as file:
            pickle.dump(features, file)

with open(os.path.join(absp, model_name, 'corrs.pkl'), 'rb') as file:
    corrs = pickle.load(file)
corrs = np.array(corrs)
corrs[np.isnan(corrs)] = 0
corrs = np.sign(corrs) * np.square(corrs)

def imresize(img, new_shape):
    dtype = img.dtype
    mult = np.abs(img).max() * 2
    m = img.astype(np.float32) / mult
    m = transform.resize(m, new_shape, order=3, mode='constant')
    m = m * mult
    return m.astype(dtype)

attacks_path = '/media/data10T_1/datasets/Voynov/__attacks__/'
attacks_generator = AttacksGenerator(os.path.join(attacks_path, get_folder_name(absp), model_name), images_are_colored, batch_size=batch_size)
a_probs_pos = np.zeros([test_generator.total, n_classes], dtype=np.float32)
a_probs_neg = np.zeros([test_generator.total, n_classes], dtype=np.float32)
pos_sizes = np.zeros([test_generator.total], dtype=np.float32)
neg_sizes = np.zeros([test_generator.total], dtype=np.float32)
for s in tqdm(range(10)):
    with open(os.path.join(absp, model_name, 'feats', 'test_feats_{}.pkl'.format(s)), 'rb') as file:
        test_feats = pickle.load(file)

    step = len(test_feats)
    print(step)
    assert step % test_generator.batch_size == 0
    
    num_batches = int(step / batch_size)
    for it in tqdm(range(num_batches)):
        X, y = test_generator[s * num_batches + it]
        X_feats = test_feats[it * batch_size: (it + 1) * batch_size]

        heat_maps = np.matmul(X_feats, corrs)
        heat_maps = np.array([imresize(hm, X[0].shape[:2]) for hm in heat_maps])
        
        heat_maps_pos = heat_maps >= 0
        heat_maps_neg = heat_maps < 0
        pos_total = heat_maps_pos.sum(axis=-1).sum(axis=-1) / np.repeat(256 * 256, len(heat_maps))
        neg_total = heat_maps_neg.sum(axis=-1).sum(axis=-1) / np.repeat(256 * 256, len(heat_maps))
        pos_sizes[(s * num_batches + it) * batch_size: (s * num_batches + it + 1) * batch_size] = pos_total
        neg_sizes[(s * num_batches + it) * batch_size: (s * num_batches + it + 1) * batch_size] = neg_total

        a_X = attacks_generator[s * num_batches + it]
        diff = a_X - X

        a_X_pos = X.copy()
        a_X_pos[heat_maps_pos] += diff[heat_maps_pos]
        a_X_neg = X.copy()
        a_X_neg[heat_maps_neg] += diff[heat_maps_neg]
        
        a_probs_pos[(s * num_batches + it) * batch_size: (s * num_batches + it + 1) * batch_size] = model.predict(a_X_pos)
        a_probs_neg[(s * num_batches + it) * batch_size: (s * num_batches + it + 1) * batch_size] = model.predict(a_X_neg)

def save_(data, fname):
    with open(os.path.join(absp, model_name, fname), 'wb') as file:
        pickle.dump(data, file)

save_(a_probs_pos, 'a_probs_pos_0.1.pkl')
save_(a_probs_neg, 'a_probs_neg_0.1.pkl')
save_(pos_sizes, 'pos_sizes.pkl')
save_(neg_sizes, 'neg_sizes.pkl')
