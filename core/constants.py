from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.mobilenet import MobileNet

MODELS = {
    "inceptionv3": InceptionV3,
    "vgg16": VGG16,
    "vgg19": VGG19,
    "xception": Xception,
    "resnet": ResNet50,
    "densenet121": DenseNet121,
    "mobilenet": MobileNet
}
TRAIN_FILE = 'train.txt'
TEST_FILE = 'test.txt'
PRED_TRAIN_FILE = 'pred_train.pkl'
PRED_TEST_FILE = 'pred_test.pkl'
