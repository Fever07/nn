import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from core.generator import Generator
from core.argparser import parse_args
from core.constants import TRAIN_FILE, TEST_FILE, MODELS

def init_and_train_model(absp, model_name, n_classes, colored, batch_size, **kwargs):
    Model_arch = MODELS[model_name]
    input_shape = [256, 256, 3 if colored else 1]
    
    # Init and compile model
    model = Model_arch(include_top=True,
                        weights=None,
                        input_shape=input_shape,
                        classes=n_classes)

    lr = 0.001
    loss = 'binary_crossentropy' if n_classes is 2 else 'categorical_crossentropy'
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=loss,
                  metrics=['accuracy'])

    # Define data generators
    abs_trainp = os.path.join(absp, TRAIN_FILE)
    abs_testp = os.path.join(absp, TEST_FILE)
    train_generator = Generator(path=abs_trainp,
                                colored=colored,
                                batch_size=batch_size,
                                num_classes=n_classes)
    test_generator = Generator(path=abs_testp,
                               colored=colored,
                               batch_size=batch_size,
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
    args = parse_args(description='Script for training of a neural network')

    if 'use_gpu' in args:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpu']
        args.pop('use_gpu')

    init_and_train_model(**args)
