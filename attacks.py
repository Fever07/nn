import os
import sys
from datetime import datetime
import numpy
from numpy.random import seed
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow import set_random_seed

from core.utils_nn import parse_file, load_gray_image, load_color_image, get_model_path, to_abs
from core.generator import ConfigurationGenerator, Generator
from core.pgd_attack import PGD_attack as pgd
import pickle

input_shape = [256, 256, 1]
n_classes = 3
batch_size = 32
colored = False
train_file = 'train.txt'
test_file = 'test.txt'
pred_train_file = 'pred_train.pkl'
pred_test_file = 'pred_test.pkl'

def attack(absp):

    seed(1337)
    set_random_seed(1337)
    rng = numpy.random.RandomState(1337)

    abs_trainp = to_abs(absp, train_file)
    abs_testp = to_abs(absp, test_file)
    abs_pred_trainp = to_abs(absp, pred_train_file)
    abs_pred_testp = to_abs(absp, pred_test_file)

    K.clear_session()
    model_path = get_model_path(absp)
    model = load_model(model_path)

    sess = K.get_session()

    generator = ConfigurationGenerator(
        abs_testp,
        abs_pred_testp,
        batch_size=batch_size,
        colored=colored
    )

    a_file_format = 'a_test_PGD_eps_{0:0.2}_max_iter_{1}.pkl'
    abs_a_file_format = to_abs(absp, a_file_format)

    max_iter=20
    eps_space = numpy.arange(0.02, 0.22, 0.02)
    # for max_eps in numpy.arange(0.02, 0.22, 0.02):
    for max_eps in eps_space:
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

        print('Attacking with eps = {0:0.2}, iters = {1} - {2}'.format(
            max_eps,
            max_iter,
            datetime.now().time().strftime('%H:%M:%S')
        ))
        
        a_probs = numpy.empty(shape=[max_iter, 0, n_classes], dtype=numpy.float32)
        for i in range(len(generator)):
            imgs, probs, labels = generator[i]

            print('Processing {0}/{1} images - {2}'.format(
                i * batch_size + len(imgs),
                generator.total,
                datetime.now().time().strftime('%H:%M:%S')
            ))

            # Attack "imgs": do "max_iter" iterations, and receive
            # generated images and probs for each iteration
            a_imgs_for_iters, a_probs_for_iters = pgd_attacker.generate_imgs_and_probs(sess, imgs, labels)
            a_probs_for_iters = numpy.array(a_probs_for_iters)

            a_probs = numpy.append(a_probs, a_probs_for_iters, axis=1)

        # Save probs to file for each iteration
        iters = numpy.arange(1, max_iter + 1)
        for it, a_probs_for_iter in zip(iters, a_probs):
            filepath = abs_a_file_format.format(max_eps, it)
            file = open(filepath, 'wb')
            pickle.dump(a_probs_for_iter, file)
            file.close()
            

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please provide path to folder')
        exit()
        
    path_to_folder = sys.argv[1]

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # d_f = '../__histology_camelyon_{}'
    # ns = [500, 1000, 2000, 5000]
    # paths = list(map(d_f.format, ns))
    attack(path_to_folder)
    # for p in paths:
    #     print(p)
    #     attack(p)
