import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy
import os
from utils import get_model_path_acc

from generator import ConfigurationGenerator

from numpy.random import seed
from tensorflow import set_random_seed
from tensorflow.keras import backend as K

seed(42)
set_random_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

if len(sys.argv) == 1:
    print('Please provide a path to folder')
    exit()
    
absp = sys.argv[1]

trainp = 'train.txt'
testp = 'test.txt'
pred_trainp = 'pred_train.txt'
pred_testp = 'pred_test.txt'
examples_folder = 'examples/'

abs_trainp = os.path.join(absp, trainp)
abs_testp = os.path.join(absp, testp)
abs_pred_trainp = os.path.join(absp, pred_trainp)
abs_pred_testp = os.path.join(absp, pred_testp)

test_gen = ConfigurationGenerator(
    abs_testp, 
    abs_pred_testp, 
    batch_size=5,
    total=10
)


model_path, acc = get_model_path_acc(absp)
model = load_model(model_path)

sess = K.get_session()
sess.run(tf.global_variables_initializer())

batch_shape=[None, 256, 256, 1]
n_classes=2
img_bounds=[0, 1]

x_input = tf.placeholder(tf.float32, shape=batch_shape)
y_input = tf.placeholder(tf.int32, shape=(batch_shape[0]))

# Loss function: the mean of the logits of the correct class
y_onehot = tf.one_hot(y_input, n_classes)
logits = model(x_input)
logits_correct_class = tf.reduce_sum(logits * y_onehot, axis=1)

loss = tf.reduce_mean(logits_correct_class)
grad = tf.gradients(loss, x_input)

rng = numpy.random.RandomState(42)

imgs, probs, labels = test_gen[0]
probs1 = model.predict(imgs, batch_size=10)
probs2 = sess.run(logits, feed_dict={x_input: imgs})
print(probs1)
print(probs2)
lgs, grads = sess.run([logits_correct_class, grad], feed_dict={x_input: at_imgs, y_input: at_lbs})
grads = grads[0]
print(numpy.linalg.norm(grads), grads.min(), grads.max())
