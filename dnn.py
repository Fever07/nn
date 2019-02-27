import os
import sys
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from core.generator import Generator
from core.utils_nn import to_abs


input_shape = [256, 256, 3]
n_classes = 4
batch_size = 32
images_are_colored = True
train_file = 'train.txt'
test_file = 'test.txt'

def init_and_train_model(absp):
    # Init and compile model
    model = InceptionV3(include_top=True,
                        weights=None,
                        input_shape=input_shape,
                        classes=n_classes)

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Define data generators
    abs_trainp = to_abs(absp, train_file)
    abs_testp = to_abs(absp, test_file)
    train_generator = Generator(path=to_abs(absp, train_file),
                                colored=images_are_colored,
                                batch_size=batch_size,
                                num_classes=n_classes)
    test_generator = Generator(path=to_abs(absp, test_file),
                               colored=images_are_colored,
                               batch_size=batch_size,
                               num_classes=n_classes)

    # Train model
    checkpoint = ModelCheckpoint(to_abs(absp, 'model_{epoch:02d}_{val_acc:.2f}.h5'),
                                 monitor='val_acc',
                                 mode='max')
    callbacks = [checkpoint]
    model.fit_generator(generator=train_generator,
                        epochs=50,
                        verbose=1,
                        validation_data=test_generator,
                        use_multiprocessing=True,
                        workers=8,
                        callbacks=callbacks)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please provide path to folder')
        exit()
        
    path_to_folder = sys.argv[1]

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    init_and_train_model(path_to_folder, use_one_gpu)