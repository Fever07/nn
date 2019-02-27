from tensorflow.keras.models import load_model
import tensorflow as tf
from generator import Generator
from utils_nn import parse_file, get_model_path, to_abs
import os
import numpy
import sys
import pickle

train_batch_size = 32
test_batch_size = 25
images_are_colored = True
n_classes = 4
train_file = 'train.txt'
test_file = 'test.txt'
pred_train_file = 'pred_train.pkl'
pred_test_file = 'pred_test.pkl'

def predict(absp):
    abs_trainp = to_abs(absp, train_file)
    abs_testp = to_abs(absp, test_file)
    abs_pred_trainp = to_abs(absp, pred_trainp)
    abs_pred_testp = to_abs(absp, pred_testp)
    
    train_paths, train_labels = parse_file(abs_trainp)
    test_paths, test_labels = parse_file(abs_testp)

    model_path = get_model_path(absp)
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
    print('Predicting train dataset...')
    train_probs = model.predict_generator(train_generator,
                                    workers=8,
                                    use_multiprocessing=True,
                                    verbose=1)

    # Calc accuracy
    pred_labels = numpy.argmax(train_probs, axis=1)
    true_labels = numpy.array(train_labels)[:len(pred_labels)]
    l = numpy.sum((pred_labels - true_labels) != 0)
    real_acc = 1 - l  / len(true_labels)
    print(real_acc)

    # Save predicted probs
    print(abs_pred_trainp)
    file = open(abs_pred_trainp, 'wb')
    pickle.dump(train_probs, file)
    print('Prediction of train dataset finished')

    # Predict test dataset
    print('Predicting test dataset...')
    test_probs = model.predict_generator(test_generator,
                                    workers=8,
                                    use_multiprocessing=True,
                                    verbose=1)

    # Calc accuracy
    pred_labels = numpy.argmax(test_probs, axis=1)
    true_labels = numpy.array(test_labels)[:len(pred_labels)]
    l = numpy.sum((pred_labels - true_labels) != 0)
    real_acc = 1 - l  / len(labels)
    print(real_acc)

    # Save predicted probs
    print(abs_pred_testp)
    file = open(abs_pred_testp, 'wb')
    pickle.dump(probs, file)
    print('Prediction of test dataset finished')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please provide path to folder')
        exit()
        
    path_to_folder = sys.argv[1]

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    predict(path_to_folder, use_one_gpu)