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

def parse_attack_files(abs_f, iter_space, eps_space):
    a_probs = []
    for max_iter in iter_space:
        for max_eps in eps_space:
            filepath = abs_f.format(max_eps, max_iter)
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

def get_model_path(absp, model_name='inceptionv3'):
    ldir = os.listdir(os.path.join(absp, model_name))
    filename_model = [p for p in ldir if p.endswith('.h5')][0]
    return os.path.join(absp, model_name, filename_model)

def load_gray_image(path):
    img = Image.open(path).resize([256, 256])
    arr = numpy.expand_dims(numpy.array(img), axis=2) / 255.0
    return arr

def load_color_image(path):
    img = Image.open(path).resize([256, 256])
    arr = numpy.array(img)[:, :, :3] / 255.0
    return arr

def get_folder_name(absp):
    if absp.endswith('/'):
        folder_name = absp.split('/')[-2]
    else:
        folder_name = absp.split('/')[-1]
    return folder_name