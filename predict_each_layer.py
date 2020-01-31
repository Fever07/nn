import os
import sys
from datetime import datetime
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow import set_random_seed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import CustomObjectScope
from functools import partial

from core.utils import parse_file, load_gray_image, load_color_image, get_model_path, get_folder_name
from core.generator import ConfigurationGenerator, Generator, AttacksGenerator
import pickle

model_name = 'mobilenet'
batch_size = 24
images_are_colored = True
n_classes = 6
test_file = 'test.txt'
pred_test_file = 'pred_test.pkl'

attacks_folder = '/media/data10T_1/datasets/Voynov/__features__/'

def get_each_layer_model(base_model):
    def parse_shape(tf_shape):
        return np.array([int(d) for d in list(tf_shape)[1:]])

    def get_shape(tf_layer):
        return parse_shape(tf_layer.shape)

    layer_names = set([type(layer).__name__ for layer in base_model.layers])
    not_used_layers = ['Concatenate', 'InputLayer']
    used_layer_names = layer_names.difference(set(not_used_layers))

    print('Using the following layer types:')
    print(used_layer_names)
    print()

    model = Model(
        inputs=base_model.input,
        outputs=[l.output for l in base_model.layers if type(l).__name__ in used_layer_names]
    )

    print('Layers total: {}'.format(len(model.output)))

    print('Total number of features: {}'.format(np.sum([np.product(s) for s in list(map(get_shape, model.output))])))

    return model

features_folder = '/media/data10T_1/datasets/Voynov/__features__'
def save_features(absp, features, i, prefix):
    folder_name = get_folder_name(absp)

    save_path = os.path.join(attacks_folder, folder_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    f = '{}_features_{}_batch_{}.pkl'.format(prefix, i, batch_size)
    abs_f = os.path.join(save_path, f)

    file = open(abs_f, 'wb')
    pickle.dump(features, file)
    file.close()


def main(absp):
    abs_testp = os.path.join(absp, test_file)
    folder_name = get_folder_name(absp)
    abs_attacks_folder = os.path.join(attacks_folder, folder_name, model_name)

    K.clear_session()
    model_path = get_model_path(absp, model_name=model_name)
    if model_name == 'mobilenet':
        with CustomObjectScope({ 'relu6': partial(tf.keras.activations.relu, max_value=6.), 
                                 'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D }):
            model = load_model(model_path)
    else:
        model = load_model(model_path)

    sess = K.get_session()

    original_generator = Generator(path=abs_testp,
                               colored=images_are_colored,
                               batch_size=batch_size,
                               num_classes=n_classes)

    each_layer_model = get_each_layer_model(model)
    print('Predicting test dataset by ---{}---, extracting features from each layer...'.format(model_name))
    # layers_features = each_layer_model.predict_generator(original_generator,
    #                                 workers=8,
    #                                 use_multiprocessing=True,
    #                                 verbose=1)

    for i in range(len(original_generator)):
        imgs, _ = original_generator[i]

        print('Processing {0}/{1} images - {2}'.format(
            i * batch_size + len(imgs),
            original_generator.total,
            datetime.now().time().strftime('%H:%M:%S')
        ))

        batch_layer_features = each_layer_model.predict(imgs)
        # save_features(absp, batch_layer_features, i, 'original')

    attacks_generator = AttacksGenerator(path=abs_attacks_folder,
                               colored=images_are_colored,
                               batch_size=batch_size)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if len(sys.argv) == 1:
        print('Please provide path to folder')
        exit()
        
    path_to_folder = sys.argv[1]
    main(path_to_folder)

    # model_names = ['inceptionv3', 'xception', 'resnet', 'densenet121']
    # for model_n in model_names:
    #     model_name = model_n
    #     predict(path_to_folder)
    # f = '../__histology_camelyon_{}'
    # ns = [500, 1000, 2000, 5000, 10000, 20000]
    # for n in ns:
    #     path_to_folder = f.format(n)
    #     predict(path_to_folder)