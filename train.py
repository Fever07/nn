import os
import sys
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from core.generator import Generator

models = {
    "inceptionv3": InceptionV3,
    "vgg16": VGG16,
    "vgg19": VGG19,
    "xception": Xception,
    "resnet": ResNet50,
    "densenet121": DenseNet121,
    "mobilenet": MobileNet
}

model_name = 'mobilenet'
input_shape = [256, 256, 1]
n_classes = 3
train_batch_size = 16
test_batch_size = 16
images_are_colored = input_shape[-1] == 3
train_file = 'train.txt'
test_file = 'test.txt'

def init_and_train_model(absp):
    Model_arch = models[model_name]
    # Init and compile model
    model = Model_arch(include_top=True,
                        weights=None,
                        input_shape=input_shape,
                        classes=n_classes)

    # model = load_model(os.path.join(absp, 'model_01_0.74.h5'))
    lr = 0.001
    loss = 'binary_crossentropy' if n_classes is 2 else 'categorical_crossentropy'
    model.compile(optimizer=tf.train.AdamOptimizer(lr),
                  loss=loss,
                  metrics=['accuracy'])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Define data generators
    abs_trainp = os.path.join(absp, train_file)
    abs_testp = os.path.join(absp, test_file)
    train_generator = Generator(path=os.path.join(absp, train_file),
                                colored=images_are_colored,
                                batch_size=train_batch_size,
                                num_classes=n_classes)
    test_generator = Generator(path=os.path.join(absp, test_file),
                               colored=images_are_colored,
                               batch_size=test_batch_size,
                               num_classes=n_classes)

    if not os.path.exists(os.path.join(absp, model_name)):
        os.mkdir(os.path.join(absp, model_name))

    # Train model
    f = 'model_{}_{{epoch:02d}}_{{val_acc:.2f}}.h5'
    checkpoint = ModelCheckpoint(os.path.join(absp, model_name, f.format(lr)),
                                 monitor='val_acc',
                                 mode='max')
    callbacks = [checkpoint]
    model.fit_generator(generator=train_generator,
                        epochs=20,
                        verbose=1,
                        validation_data=test_generator,
                        use_multiprocessing=True,
                        workers=4,
                        callbacks=callbacks)

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if len(sys.argv) == 1:
        print('Please provide path to folder')
        exit()
        
    path_to_folder = sys.argv[1]
    init_and_train_model(path_to_folder)

    # f = '../__histology_camelyon_{}'
    # ns = [50000, 100000]
    # for n in ns:
    #     path_to_folder = f.format(n)
    #     init_and_train_model(path_to_folder)
