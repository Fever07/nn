import os
import sys
from datetime import datetime
import numpy as np
from numpy.random import seed
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
# required for mobilenet
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras import backend as K
from tensorflow import set_random_seed

from core.utils import parse_file, load_gray_image, load_color_image, get_model_path, get_folder_name
from core.generator import ConfigurationGenerator, Generator
from core.deepfool_attack import deepfool_attack
from core.constants import TRAIN_FILE, TEST_FILE, PRED_TRAIN_FILE, PRED_TEST_FILE
from core.argparser import parse_args
import pickle
from functools import partial
from tqdm import tqdm

def calc_l2norm(v):
    n = np.linalg.norm(v, axis=1)
    n = np.linalg.norm(n, axis=1)
    n = np.linalg.norm(n, axis=1)
    return n

def calc_linfnorm(v):
    n = np.max(np.abs(v), axis=(1, 2, 3))
    return n

def softmax(v):
    return np.exp(v) / np.sum(np.exp(v))

def attack_deepfool(absp, model_name, n_classes, colored, batch_size, **kwargs):

    seed(1337)
    set_random_seed(1337)
    rng = np.random.RandomState(1337)

    abs_trainp = os.path.join(absp, TRAIN_FILE)
    abs_testp = os.path.join(absp, TEST_FILE)
    abs_pred_trainp = os.path.join(absp, model_name, PRED_TRAIN_FILE)
    abs_pred_testp = os.path.join(absp, model_name, PRED_TEST_FILE)
    input_shape = [256, 256, 3 if colored else 1]

    K.clear_session()
    sess = K.get_session()
    model_path = get_model_path(absp, model_name=model_name)
    if model_name == 'mobilenet':
        with CustomObjectScope({ 'relu6': partial(tf.keras.activations.relu, max_value=6.), 
                                 'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D }):
            model = load_model(model_path)
    else:
        model = load_model(model_path)

    model.layers[-1].activation = tf.keras.activations.linear
    model.save('__temp_model__')
    model = load_model('__temp_model__')
    os.remove('__temp_model__')

    generator = Generator(
        abs_testp,
        batch_size=batch_size,
        colored=colored,
        num_classes=n_classes
    )

    a_file_format = 'a_test_DF_probs_max_iter_{}.pkl'
    abs_a_file_format = os.path.join(absp, model_name, a_file_format)
    a_file_l2norm_format = 'a_test_DF_l2norms_max_iter_{}.pkl'
    abs_a_file_l2norm_format = os.path.join(absp, model_name, a_file_l2norm_format)
    a_file_linfnorm_format = 'a_test_DF_linfnorms_max_iter_{}.pkl'
    abs_a_file_linfnorm_format = os.path.join(absp, model_name, a_file_linfnorm_format)

    print('Attacking of %s' % model_name)

    max_iter = 50

    a_probs = np.zeros(shape=[generator.total, n_classes], dtype=np.float32)
    diff_l2norms = np.zeros(shape=[generator.total], dtype=np.float32)
    diff_linfnorms = np.zeros(shape=[generator.total], dtype=np.float32)
    grads = [tf.gradients(model.output[:, i], model.input)[0] for i in range(n_classes)]
    grads = tf.stack(grads)
    grads = tf.transpose(grads, perm=[1, 0, 2, 3, 4])
    for i in tqdm(range(len(generator))):
        imgs, categorical_labels = generator[i]
        labels = np.argmax(categorical_labels, axis=1)

        a_imgs = deepfool_attack(sess=sess,
                                x=model.input,
                                predictions=model.output,
                                grads=grads,
                                imgs=imgs,
                                n_classes=n_classes,
                                overshoot=0.02,
                                max_iter=50,
                                clip_min=0.0,
                                clip_max=1.0)

        diff_l2norms_batch = calc_l2norm(imgs - a_imgs)
        diff_l2norms[i * batch_size: (i + 1) * batch_size] = diff_l2norms_batch

        diff_linfnorms_batch = calc_linfnorm(imgs - a_imgs)
        diff_linfnorms[i * batch_size: (i + 1) * batch_size] = diff_linfnorms_batch

        a_probs_batch = np.array([softmax(p) for p in model.predict(a_imgs)])
        a_probs[i * batch_size: (i + 1) * batch_size] = a_probs_batch

    fpath = abs_a_file_format.format(max_iter)
    file = open(fpath, 'wb')
    pickle.dump(a_probs, file)
    file.close()

    fpath = abs_a_file_l2norm_format.format(max_iter)
    file = open(fpath, 'wb')
    pickle.dump(diff_l2norms, file)
    file.close()
    
    fpath = abs_a_file_linfnorm_format.format(max_iter)
    file = open(fpath, 'wb')
    pickle.dump(diff_linfnorms, file)
    file.close()
    

if __name__ == '__main__':
    args = parse_args(description='Deepfool attacks')

    if 'use_gpu' in args:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpu']
        args.pop('use_gpu')

    attack_deepfool(**args)
