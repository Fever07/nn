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

from core.utils_nn import parse_file, load_gray_image, load_color_image, get_model_path, get_folder_name
from core.generator import ConfigurationGenerator, Generator
from core.pgd_attack import PGD_attack as pgd
import pickle
from functools import partial
from tqdm import tqdm

from cleverhans import attacks_tf
from cleverhans import model as  clm

model_name = 'inceptionv3'
input_shape = [256, 256, 1]
n_classes = 2
batch_size = 20
colored = input_shape[-1] == 3
train_file = 'train.txt'
test_file = 'test.txt'
pred_train_file = 'pred_train.pkl'
pred_test_file = 'pred_test.pkl'

def calc_l2norm(v):
    n = np.linalg.norm(v, axis=1)
    n = np.linalg.norm(n, axis=1)
    n = np.linalg.norm(n, axis=1)
    return n

def calc_linfnorm(v):
    n = np.max(np.abs(v), axis=(1, 2, 3))
    return n

def attack(absp):

    seed(1337)
    set_random_seed(1337)
    rng = np.random.RandomState(1337)

    abs_trainp = os.path.join(absp, train_file)
    abs_testp = os.path.join(absp, test_file)
    abs_pred_trainp = os.path.join(absp, model_name, pred_train_file)
    abs_pred_testp = os.path.join(absp, model_name, pred_test_file)

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
        colored=colored
    )

    a_file_format = 'a_test_CW_probs_conf_{0}_lr_{1}_c_{2}_max_iter_{3}.pkl'
    abs_a_file_format = os.path.join(absp, model_name, a_file_format)
    a_file_l2norm_format = 'a_test_CW_l2norms_conf_{0}_lr_{1}_c_{2}_max_iter_{3}.pkl'
    abs_a_file_l2norm_format = os.path.join(absp, model_name, a_file_l2norm_format)
    a_file_linfnorm_format = 'a_test_CW_linfnorms_conf_{0}_lr_{1}_c_{2}_max_iter_{3}.pkl'
    abs_a_file_linfnorm_format = os.path.join(absp, model_name, a_file_linfnorm_format)

    print('Attacking of %s' % model_name)

    confidence = 0
    lr = 0.01
    c = 1
    max_iter = 500
    wrapped_model = clm.CallableModelWrapper(model, 'logits')
    cw_attacker = attacks_tf.CarliniWagnerL2(sess=sess, 
                                            model=wrapped_model,
                                            batch_size=batch_size,
                                            confidence=confidence,
                                            targeted=False,
                                            learning_rate=lr,
                                            binary_search_steps=5,
                                            max_iterations=max_iter,
                                            abort_early=True,
                                            initial_const=c,
                                            clip_min=0,
                                            clip_max=1,
                                            num_labels=n_classes,
                                            shape=input_shape)
    
    a_probs = np.zeros(shape=[generator.total, n_classes], dtype=np.float32)
    diff_l2norms = np.zeros(shape=[generator.total], dtype=np.float32)
    diff_linfnorms = np.zeros(shape=[generator.total], dtype=np.float32)
    for i in tqdm(range(len(generator))):
        imgs, categorical_labels = generator[i]

        a_imgs = cw_attacker.attack_batch(imgs, categorical_labels)
        diff_l2norms_batch = calc_l2norm(imgs - a_imgs)
        diff_l2norms[i * batch_size: (i + 1) * batch_size] = diff_l2norms_batch

        diff_linfnorms_batch = calc_linfnorm(imgs - a_imgs)
        diff_linfnorms[i * batch_size: (i + 1) * batch_size] = diff_linfnorms_batch

        a_probs_batch = model.predict(a_imgs)
        a_probs[i * batch_size: (i + 1) * batch_size] = a_probs_batch

        probs = model.predict(imgs)

    fpath = abs_a_file_format.format(confidence, lr, c, max_iter)
    file = open(fpath, 'wb')
    pickle.dump(a_probs, file)
    file.close()

    fpath = abs_a_file_l2norm_format.format(confidence, lr, c, max_iter)
    file = open(fpath, 'wb')
    pickle.dump(diff_l2norms, file)
    file.close()
    
    fpath = abs_a_file_linfnorm_format.format(confidence, lr, c, max_iter)
    file = open(fpath, 'wb')
    pickle.dump(diff_linfnorms, file)
    file.close()
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if len(sys.argv) == 1:
        print('Please provide path to folder')
        exit()
        
    path_to_folder = sys.argv[1]
    attack(path_to_folder)
