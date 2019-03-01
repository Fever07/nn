import os
from parse import parse
from PIL import Image
import numpy
import pickle

def parse_file(filepath):
    def parse_orig_line(line):
        line_format = '{0} {1}'
        path, label = parse(line_format, line)
        label = int(label)
        return path, label

    file = open(filepath, 'r')
    lines = file.readlines()
    parsed = [parse_orig_line(line) for line in lines]
    paths_labels = list(zip(*parsed))
    file.close()
    return paths_labels

def parse_pred_file(filepath):
    file = open(filepath, 'rb')
    probs = pickle.load(file)
    file.close()
    return probs

def parse_attack_files(absp, iter_space, eps_space, test=True):
    if test:
        f = 'a_test_PGD_eps_{0:0.2}_max_iter_{1}.pkl'
    else:
        f = 'a_train_PGD_eps_{0:0.2}_max_iter_{1}.pkl'

    a_probs = []
    for max_iter in iter_space:
        for max_eps in eps_space:
            filename = f.format(max_eps, max_iter)
            filepath = os.path.join(absp, filename)
            file = open(filepath, 'rb')
            a_iter_eps_probs = pickle.load(file)
            file.close()

            a_probs.append(a_iter_eps_probs)

    a_probs = numpy.array(a_probs)
    shape = a_probs.shape
    a_probs = a_probs.reshape(len(iter_space), len(eps_space), *shape[1: ])
    print(a_probs.shape, a_probs.dtype)
    return a_probs

def compose_file(filepath, paths, probs):
    def compose_line(path, prob):
        f = '{0:0.3}'
        str_prob = list(map(f.format, prob))
        str_prob = ' '.join(str_prob)
        resf = '{0} {1}\n'
        return resf.format(path, str_prob)

    file = open(filepath, 'w')
    paths_probs = list(zip(*[paths, probs]))
    lines = [compose_line(*tup) for tup in paths_probs]
    file.writelines(lines)
    file.close()

def get_model_path(absp):
    ldir = os.listdir(absp)
    filename_model = [p for p in ldir if p.endswith('.h5')][0]
    return to_abs(absp, filename_model)

def load_gray_image(path):
    img = Image.open(path).resize([256, 256])
    arr = numpy.expand_dims(numpy.array(img), axis=2) / 255.0
    return arr

def load_color_image(path):
    img = Image.open(path)
    arr = numpy.array(img)[:, :, :3] / 255.0
    return arr

def to_abs(absp, path):
    return os.path.join(absp, path)