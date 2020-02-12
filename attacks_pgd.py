import os
import sys
from datetime import datetime
import numpy
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

attacks_folder = '/media/data10T_1/datasets/Voynov/__attacks__'
def save_attacks(absp, a_imgs, i):
    folder_name = get_folder_name(absp)

    save_path = os.path.join(attacks_folder, folder_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    f = 'attacks_{}_batch_{}.pkl'.format(i, batch_size)
    abs_f = os.path.join(save_path, f)

    file = open(abs_f, 'wb')
    pickle.dump(a_imgs, file)
    file.close()


def attack(absp, model_name, n_classes, colored, batch_size, **kwargs):

    seed(1337)
    set_random_seed(1337)
    rng = numpy.random.RandomState(1337)

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

    generator = ConfigurationGenerator(
        abs_testp,
        abs_pred_testp,
        batch_size=batch_size,
        colored=colored
    )

    a_file_format = 'a_test_PGD_eps_{0:0.2}_max_iter_{1}.pkl'
    abs_a_file_format = os.path.join(absp, a_file_format)

    max_iter=20
    eps_space = numpy.arange(0.10, 0.12, 0.02)
    print(eps_space)
    print('Attacking of %s' % model_name)
    # for max_eps in numpy.arange(0.02, 0.22, 0.02):
    for max_eps in tqdm(eps_space):
        pgd_attacker = pgd(model,
                           batch_shape=[None]+input_shape,
                           max_epsilon=max_eps,
                           max_iter=max_iter,
                           initial_lr=1,
                           lr_decay=1,
                           targeted=False,
                           n_classes=n_classes,
                           img_bounds=[0, 1],
                           rng=rng)

        # print('Attacking with eps = {0:0.2}, iters = {1} - {2}'.format(
        #     max_eps,
        #     max_iter,
        #     datetime.now().time().strftime('%H:%M:%S')
        # ))
        
        a_probs = numpy.empty(shape=[max_iter, 0, n_classes], dtype=numpy.float32)
        for i in tqdm(range(len(generator))):
            imgs, probs, labels = generator[i]

            # print('Processing {0}/{1} images - {2}'.format(
            #     i * batch_size + len(imgs),
            #     generator.total,
            #     datetime.now().time().strftime('%H:%M:%S')
            # ))

            # Attack "imgs": do "max_iter" iterations, and receive
            # generated images and probs for each iteration
            # a_imgs_for_iters, a_probs_for_iters = pgd_attacker.generate_imgs_and_probs(sess, imgs, labels)
            # a_probs_for_iters = numpy.array(a_probs_for_iters)

            # a_probs = numpy.append(a_probs, a_probs_for_iters, axis=1)

            a_imgs = pgd_attacker.generate(sess, imgs, labels)
            save_attacks(absp, a_imgs, i)

        # Save probs to file for each iteration
        # iters = numpy.arange(1, max_iter + 1)
        # for it, a_probs_for_iter in zip(iters, a_probs):
        #     filepath = abs_a_file_format.format(max_eps, it)
        #     file = open(filepath, 'wb')
        #     pickle.dump(a_probs_for_iter, file)
        #     file.close()
            

if __name__ == '__main__':
    args = parse_args(description='PGD attacks')

    if 'use_gpu' in args:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpu']
        args.pop('use_gpu')

    attack_lbfgs(**args)
