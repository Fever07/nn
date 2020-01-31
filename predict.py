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
from functools import partial

model_name = 'resnet'
train_batch_size = 16
test_batch_size = 16
images_are_colored = True
n_classes = 2
train_file = 'train.txt'
test_file = 'test.txt'
pred_train_file = 'pred_train.pkl'
pred_test_file = 'pred_test.pkl'

def predict(absp):
    abs_trainp = os.path.join(absp, train_file)
    abs_testp = os.path.join(absp, test_file)
    abs_pred_trainp = os.path.join(absp, model_name, pred_train_file)
    abs_pred_testp = os.path.join(absp, model_name, pred_test_file)
    
    train_paths, train_labels = parse_file(abs_trainp)
    test_paths, test_labels = parse_file(abs_testp)

    model_path = get_model_path(absp, model_name=model_name)
    if model_name == 'mobilenet':
        with CustomObjectScope({ 'relu6': partial(tf.keras.activations.relu, max_value=6.), 
                                 'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D }):
            model = load_model(model_path)
    else:
        model = load_model(model_path)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_generator = Generator(path=abs_trainp,
                                colored=images_are_colored,
                                batch_size=train_batch_size,
                                num_classes=n_classes)
    test_generator = Generator(path=abs_testp,
                               colored=images_are_colored,
                               batch_size=test_batch_size,
                               num_classes=n_classes)

    # Predict train dataset
    # print('Predicting train dataset...')
    # train_probs = model.predict_generator(train_generator,
    #                                 workers=8,
    #                                 use_multiprocessing=True,
    #                                 verbose=1)

    # # Calc accuracy
    # pred_labels = numpy.argmax(train_probs, axis=1)
    # true_labels = numpy.array(train_labels)[:len(pred_labels)]
    # l = numpy.sum((pred_labels - true_labels) != 0)
    # real_acc = 1 - l  / len(true_labels)
    # print(real_acc)

    # # Save predicted probs
    # print(abs_pred_trainp)
    # file = open(abs_pred_trainp, 'wb')
    # pickle.dump(train_probs, file)
    # print('Prediction of train dataset finished')

    # Predict test dataset
    print('Predicting test dataset by {}...'.format(model_name))
    test_probs = model.predict_generator(test_generator,
                                    workers=8,
                                    use_multiprocessing=True,
                                    verbose=1)

    # Calc accuracy
    pred_labels = numpy.argmax(test_probs, axis=1)
    true_labels = numpy.array(test_labels)[:len(pred_labels)]
    l = numpy.sum((pred_labels - true_labels) != 0)
    real_acc = 1 - l  / len(true_labels)
    print(real_acc)

    # Save predicted probs
    print(abs_pred_testp)
    file = open(abs_pred_testp, 'wb')
    pickle.dump(test_probs, file)
    file.close()
    print('Prediction of test dataset by {} finished'.format(model_name))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if len(sys.argv) == 1:
        print('Please provide path to folder')
        exit()
        
    path_to_folder = sys.argv[1]
    predict(path_to_folder)

    # model_names = ['inceptionv3', 'xception', 'resnet', 'densenet121']
    # for model_n in model_names:
    #     model_name = model_n
    #     predict(path_to_folder)
    # f = '../__histology_camelyon_{}'
    # ns = [500, 1000, 2000, 5000, 10000, 20000]
    # for n in ns:
    #     path_to_folder = f.format(n)
    #     predict(path_to_folder)