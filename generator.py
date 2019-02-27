import numpy
from tensorflow import keras
from PIL import Image
from utils_nn import parse_file, parse_pred_file, load_gray_image, load_color_image

class AbstractGenerator(keras.utils.Sequence):
    def __init__(self, path, colored, batch_size=32, shape=1):
        self.path = path
        self.batch_size = batch_size
        self.colored = colored
        self.load_image = load_color_image if colored else load_gray_image

    def __data_generation(self, start_batch):
        pass

    def __len__(self):
        return int(numpy.ceil(self.total / self.batch_size))

    def __getitem__(self, index):
        pass


class Generator(AbstractGenerator):
    def __init__(self, path, colored=False, batch_size=32, num_classes=2):
        AbstractGenerator.__init__(self, path, colored, batch_size)
        self.paths, self.labels = parse_file(self.path)
        self.total = len(self.paths)
        self.num_classes = num_classes

    def __data_generation(self, start_batch):
        end_batch = min(start_batch + self.batch_size, self.total)
        labels = self.labels[start_batch: end_batch]
        data = []
        for i in range(start_batch, end_batch):
            arr = self.load_image(self.paths[i])
            data.append(arr)

        return numpy.array(data), keras.utils.to_categorical(labels, num_classes=self.num_classes)

    def __getitem__(self, index):
        x, y = self.__data_generation(index * self.batch_size)
        return x, y


class ConfigurationGenerator(AbstractGenerator):
    def __init__(self, orig_path, pred_path, colored=False, batch_size=32, total=None):
        AbstractGenerator.__init__(self, pred_path, colored, batch_size)
        self.paths, self.labels = parse_file(orig_path)
        self.probs = parse_pred_file(pred_path)
        self.num_classes = len(self.probs[0])
        if total is not None:
            self.total = total
        else:
            self.total = len(self.paths)

    def __data_generation(self, start_batch):
        end_batch = min(start_batch + self.batch_size, self.total)
        probs = self.probs[start_batch: end_batch]
        labels = self.labels[start_batch: end_batch]
        data = []
        for i in range(start_batch, end_batch):
            arr = self.load_image(self.paths[i])
            data.append(arr)

        return numpy.array(data), numpy.array(probs), numpy.array(labels)

    def __getitem__(self, index):
        imgs, probs, labels = self.__data_generation(index * self.batch_size)
        return imgs, probs, labels 
