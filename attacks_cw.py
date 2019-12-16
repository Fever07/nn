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

def attack(absp):

    seed(1337)
    set_random_seed(1337)
    rng = numpy.random.RandomState(1337)

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
        colored=colored,
        total=batch_size * 5
    )

    a_file_format = 'a_test_CW_eps_{0:0.2}_max_iter_{1}.pkl'
    abs_a_file_format = os.path.join(absp, model_name, a_file_format)

    max_iter=200
    print('Attacking of %s' % model_name)
    wrapped_model = clm.CallableModelWrapper(model, 'logits')
    cw_attacker = attacks_tf.CarliniWagnerL2(sess=sess, 
                                            model=wrapped_model,
                                            batch_size=batch_size,
                                            confidence=0,
                                            targeted=False,
                                            learning_rate=0.01,
                                            binary_search_steps=5,
                                            max_iterations=max_iter,
                                            abort_early=True,
                                            initial_const=1,
                                            clip_min=0,
                                            clip_max=1,
                                            num_labels=n_classes,
                                            shape=input_shape)
    
    # a_probs = numpy.empty(shape=[max_iter, 0, n_classes], dtype=numpy.float32)
    for i in tqdm(range(len(generator))):
        imgs, categorical_labels = generator[i]

        a_imgs = cw_attacker.attack_batch(imgs, categorical_labels)
        probs = model.predict(imgs)
        a_probs = model.predict(a_imgs)
        print(probs, a_probs)

    # Save probs to file for each iteration
    # iters = numpy.arange(1, max_iter + 1)
    # for it, a_probs_for_iter in zip(iters, a_probs):
    #     filepath = abs_a_file_format.format(max_eps, it)
    #     file = open(filepath, 'wb')
    #     pickle.dump(a_probs_for_iter, file)
    #     file.close()
            

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if len(sys.argv) == 1:
        print('Please provide path to folder')
        exit()
        
    path_to_folder = sys.argv[1]
    attack(path_to_folder)

