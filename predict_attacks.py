import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from core.generator import AttacksGenerator
from core.utils import parse_file, parse_pred_file, get_model_path, get_folder_name

from tensorflow.keras.utils import CustomObjectScope
from functools import partial

target_model_name = 'inceptionv3'
attacks_model_name = 'inceptionv3'
test_batch_size = 20
images_are_colored = False
n_classes = 2
test_file = 'test.txt'
pred_test_file = 'pred_test.pkl'

attacks_folder = '/media/data10T_1/datasets/Voynov/__attacks__'
def predict_attacks(absp):
    abs_testp = os.path.join(absp, test_file)
    abs_pred_testp = os.path.join(absp, target_model_name, pred_test_file)
    a_pred_test_file = '{}_on_{}.pkl'.format(target_model_name, attacks_model_name)
    abs_a_pred_testp = os.path.join(absp, target_model_name, a_pred_test_file)
    folder_name = get_folder_name(absp)
    abs_attacks_folder = os.path.join(attacks_folder, folder_name, attacks_model_name)

    K.clear_session()
    model_path = get_model_path(absp, model_name=target_model_name)
    if target_model_name == 'mobilenet':
        with CustomObjectScope({ 'relu6': partial(tf.keras.activations.relu, max_value=6.), 
                                 'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D }):
            model = load_model(model_path)
    else:
        model = load_model(model_path)

    sess = K.get_session()

    attacks_generator = AttacksGenerator(path=abs_attacks_folder,
                               colored=images_are_colored,
                               batch_size=test_batch_size)

    # Predict attacks, generated on 'attacks_model' from test dataset
    print('Predicting by ---{}--- attacks of ---{}--- of test dataset...'.format(target_model_name, attacks_model_name))
    a_probs = model.predict_generator(attacks_generator,
                                    workers=8,
                                    use_multiprocessing=True,
                                    verbose=1)

    # Calc rate of errors
    _, true_labels = parse_file(abs_testp)
    pred_probs = parse_pred_file(abs_pred_testp)
    pred_labels = np.argmax(pred_probs, axis=1)

    correct_inds = np.where(true_labels - pred_labels == 0)[0]
    cor_pred_labels = pred_labels[correct_inds]
    cor_a_probs = a_probs[correct_inds]
    cor_a_labels = np.argmax(cor_a_probs, axis=1)
    error_len = len(np.where(cor_a_labels - cor_pred_labels != 0)[0])
    error_rate = error_len / len(correct_inds)
    print('Error rate: {}'.format(error_rate))

    # Save predicted probs
    print('Saving to a file %s' % abs_a_pred_testp)
    file = open(abs_a_pred_testp, 'wb')
    pickle.dump(a_probs, file)
    file.close()
    print('Prediction of pre-generated attacks finished')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if len(sys.argv) == 1:
        print('Please provide path to folder')
        exit()
        
    path_to_folder = sys.argv[1]
    predict_attacks(path_to_folder)

    # model_names = ['inceptionv3', 'xception', 'resnet', 'densenet121', 'mobilenet']
    
    # new_model = 'mobilenet'
    # attacks_model_name = new_model
    # for used_model in model_names:
    #     # if new_model != used_model:
    #     target_model_name = used_model
    #     predict_attacks(path_to_folder)

    # target_model_name = new_model
    # for used_model in model_names[1:]:
    #     attacks_model_name = used_model
    #     predict_attacks(path_to_folder)

    # for model_name_1 in model_names:
    #     for model_name_2 in model_names:
    #         target_model_name = model_name_1
    #         attacks_model_name = model_name_2
    #         predict_attacks(path_to_folder)
