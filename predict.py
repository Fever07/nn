import os
import sys
import numpy
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
# required for mobilenet
from tensorflow.keras.utils import CustomObjectScope

from core.generator import Generator
from core.utils import parse_file, get_model_path
from core.constants import TRAIN_FILE, TEST_FILE, PRED_TRAIN_FILE, PRED_TEST_FILE
from core.argparser import parse_args
from functools import partial


def predict_generator(model_name, model_path, generator, labels, abs_pred_p):
    if model_name == 'mobilenet':
        with CustomObjectScope({ 'relu6': partial(tf.keras.activations.relu, max_value=6.), 
                                 'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D }):
            model = load_model(model_path)
    else:
        model = load_model(model_path)

    probs = model.predict(generator,
                            workers=8,
                            use_multiprocessing=True,
                            verbose=1)

    # Calc accuracy
    pred_labels = numpy.argmax(probs, axis=1)
    true_labels = numpy.array(labels)[:len(pred_labels)]
    l = numpy.sum((pred_labels - true_labels) != 0)
    real_acc = 1 - l  / len(true_labels)
    print(real_acc)

    # Save predicted probs
    print(abs_pred_p)
    file = open(abs_pred_p, 'wb')
    pickle.dump(probs, file)


def predict_train_and_test(
    absp, 
    model_name, 
    n_classes, 
    colored, 
    batch_size, 
    predict_train=True, 
    predict_test=True, 
    **kwargs):

    model_path = get_model_path(absp, model_name=model_name)
    if predict_train:
        abs_trainp = os.path.join(absp, TRAIN_FILE)
        abs_pred_trainp = os.path.join(absp, model_name, PRED_TRAIN_FILE)

        _, train_labels = parse_file(abs_trainp)
        train_generator = Generator(path=abs_trainp,
                                colored=colored,
                                batch_size=batch_size,
                                num_classes=n_classes)

        print('Predicting train dataset...')
        predict_generator(model_name, model_path, train_generator, train_labels, abs_pred_trainp)

    if predict_test:
        abs_testp = os.path.join(absp, TEST_FILE)
        abs_pred_testp = os.path.join(absp, model_name, PRED_TEST_FILE)

        _, test_labels = parse_file(abs_testp)
        test_generator = Generator(path=abs_testp,
                               colored=colored,
                               batch_size=batch_size,
                               num_classes=n_classes)

        print('Predicting test dataset...')
        predict_generator(model_name, model_path, test_generator, test_labels, abs_pred_testp)


if __name__ == '__main__':
    args = parse_args(description='Script for predicting a dataset')

    if 'use_gpu' in args:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpu']
        args.pop('use_gpu')

    predict_train_and_test(predict_train=False, **args)
