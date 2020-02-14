import numpy
from tensorflow import keras
from PIL import Image
import pickle
from parse import parse
import os

from core.utils import parse_file, parse_pred_file, load_gray_image, load_color_image, detect_dataset_configuration

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
    def __init__(self, path, colored=False, batch_size=32, num_classes=2, total=None):
        AbstractGenerator.__init__(self, path, colored, batch_size)
        self.paths, self.labels = parse_file(self.path)
        self.colored, self.num_classes, self.batch_size = detect_dataset_configuration(self.paths, self.labels)
        if total is not None:
            self.total = total
        else:
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
        self.colored, self.num_classes, self.batch_size = detect_dataset_configuration(self.paths, self.labels)
        self.probs = parse_pred_file(pred_path)
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


class AttacksGenerator(keras.utils.Sequence):
    def __init__(self, path, colored, batch_size=32):
        self.path = path
        self.batch_size = batch_size
        self.colored = colored

        ldir = os.listdir(path)
        self.paths = [os.path.join(path, p) for p in ldir]

        f = 'attacks_{}_batch_{}.pkl'
        self.abs_f = os.path.join(path, f)
        self.step_size = int(parse(f, ldir[0])[1])
        self.max_i = len(self.paths)
        self.total = self.max_i * self.step_size

    def __data_generation(self, start_batch):
        end_batch = min(start_batch + self.batch_size, self.total) - 1

        step_size = self.step_size
        start_i = start_batch // step_size
        end_i = end_batch // step_size

        start_step = start_i * step_size
        start_rel_batch = start_batch - start_step
        end_rel_batch = end_batch - start_step

        len_i = end_i - start_i + 1
        channels = 3 if self.colored else 1
        attacks = numpy.zeros(shape=[step_size * len_i, 256, 256, channels], dtype=numpy.float64)

        for i in range(0, len_i):
            fp = self.abs_f.format(i + start_i, self.step_size)
            file = open(fp, 'rb')
            step_attacks = pickle.load(file)
            file.close()

            if len(step_attacks) != step_size:
                print('Attacks file is corrupted, i = {}'.format(i))
            attacks[i * step_size: (i + 1) * step_size] = step_attacks

        batch_attacks = attacks[start_rel_batch: end_rel_batch + 1]
        return batch_attacks


    def __len__(self):
        return int(numpy.ceil(self.total / self.batch_size))

    def __getitem__(self, index):
        a_imgs = self.__data_generation(index * self.batch_size)
        return a_imgs 
