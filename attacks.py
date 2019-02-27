from pgd_attack import PGD_attack as pgd
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from PIL import Image
import os
import numpy
from adversarial_examples.step_pgd_attack import step_pgd_attack as spa
from adversarial_examples.box_constrained_attack import box_constrained_attack as bca

from numpy.random import seed
from tensorflow import set_random_seed
from utils_nn import parse_file, load_gray_image, load_color_image, get_model_path
from attack_generator import attack_generator, attack_iterative_generator
from generator import ConfigurationGenerator, Generator
import sys
import json
from datetime import datetime

input_shape = [256, 256, 3]
n_classes = 4
batch_size = 32
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

    model_path = get_model_path(absp)
    model = load_model(model_path)

    sess = K.get_session()

    generator = ConfigurationGenerator(
        abs_testp,
        abs_pred_testp,
        batch_size=batch_size,
        colored=images_are_colored
    )

    a_file_format = 'a_test_PGD_eps_{0:0.2}_max_iter_{{}}.pkl'
    abs_a_file_format = to_abs(absp, a_file_format)

    max_iter=20
    for max_eps in numpy.arange(0.02, 0.22, 0.02):
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
        attack_iterative_generator(
            sess,
            model,
            generator,
            pgd_attacker,
            abs_a_file_format.format(max_eps)
        )

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please provide path to folder')
        exit()
        
    path_to_folder = sys.argv[1]

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    attack(path_to_folder)
