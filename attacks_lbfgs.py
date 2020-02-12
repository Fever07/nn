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
from core.pgd_attack import PGD_attack as pgd
from core.constants import TRAIN_FILE, TEST_FILE, PRED_TRAIN_FILE, PRED_TEST_FILE
import pickle
from functools import partial
from tqdm import tqdm

from cleverhans import attacks_tf
from cleverhans import model as  clm

def calc_l2norm(v):
    n = np.linalg.norm(v, axis=1)
    n = np.linalg.norm(n, axis=1)
    n = np.linalg.norm(n, axis=1)
    return n

def calc_linfnorm(v):
    n = np.max(np.abs(v), axis=(1, 2, 3))
    return n

def attack_lbfgs(absp, model_name, n_classes, colored, batch_size, **kwargs):

    seed(1337)
    set_random_seed(1337)
    rng = np.random.RandomState(1337)

    abs_trainp = os.path.join(absp, TRAIN_FILE)
    abs_testp = os.path.join(absp, TEST_FILE)
    abs_pred_trainp = os.path.join(absp, model_name, PRED_TRAIN_FILE)
    abs_pred_testp = os.path.join(absp, model_name, PRED_TEST_FILE)
    input_shape = [256, 256, 3 if colored else 1]

    K.clear_session()
    model_path = get_model_path(absp, model_name=model_name)
    if model_name == 'mobilenet':
        with CustomObjectScope({ 'relu6': partial(tf.keras.activations.relu, max_value=6.), 
                                 'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D }):
            model = load_model(model_path)
    else:
        model = load_model(model_path)

    sess = K.get_session()

    generator = Generator(
        abs_testp,
        batch_size=batch_size,
        colored=colored,
        num_classes=n_classes
    )

    a_file_format = 'a_test_LBFGS_probs_c_{2}_max_iter_{3}.pkl'
    abs_a_file_format = os.path.join(absp, model_name, a_file_format)
    a_file_l2norm_format = 'a_test_LBFGS_l2norms_c_{2}_max_iter_{3}.pkl'
    abs_a_file_l2norm_format = os.path.join(absp, model_name, a_file_l2norm_format)
    a_file_linfnorm_format = 'a_test_LBFGS_linfnorms_c_{2}_max_iter_{3}.pkl'
    abs_a_file_linfnorm_format = os.path.join(absp, model_name, a_file_linfnorm_format)

    print('Attacking of %s' % model_name)

    c = 1
    max_iter = 50
    targeted_label = tf.Variable(dtype=np.float32, shape=[batch_size, n_classes], initial_value=np.zeros([batch_size, n_classes]))
    lbfgs_attacker = attacks_tf.LBFGS_attack(sess=sess,
                                            x=model.input,
                                            model_preds=model.output,
                                            targeted_label=targeted_label,
                                            binary_search_steps=5,
                                            max_iterations=max_iter,
                                            initial_const=c,
                                            clip_min=0,
                                            clip_max=1,
                                            nb_classes=n_classes,
                                            batch_size=batch_size)
    
    a_probs = np.zeros(shape=[generator.total, n_classes], dtype=np.float32)
    diff_l2norms = np.zeros(shape=[generator.total], dtype=np.float32)
    diff_linfnorms = np.zeros(shape=[generator.total], dtype=np.float32)
    for i in tqdm(range(len(generator))):
        imgs, categorical_labels = generator[i]

        a_imgs = lbfgs_attacker.attack(imgs, categorical_labels)
        diff_l2norms_batch = calc_l2norm(imgs - a_imgs)
        diff_l2norms[i * batch_size: (i + 1) * batch_size] = diff_l2norms_batch

        diff_linfnorms_batch = calc_linfnorm(imgs - a_imgs)
        diff_linfnorms[i * batch_size: (i + 1) * batch_size] = diff_linfnorms_batch

        a_probs_batch = model.predict(a_imgs)
        a_probs[i * batch_size: (i + 1) * batch_size] = a_probs_batch

    fpath = abs_a_file_format.format(c, max_iter)
    file = open(fpath, 'wb')
    pickle.dump(a_probs, file)
    file.close()

    fpath = abs_a_file_l2norm_format.format(c, max_iter)
    file = open(fpath, 'wb')
    pickle.dump(diff_l2norms, file)
    file.close()
    
    fpath = abs_a_file_linfnorm_format.format(c, max_iter)
    file = open(fpath, 'wb')
    pickle.dump(diff_linfnorms, file)
    file.close()
    

if __name__ == '__main__':
    args = parse_args(description='LBFGS attacks')

    if 'use_gpu' in args:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpu']
        args.pop('use_gpu')

    attack_lbfgs(**args)
