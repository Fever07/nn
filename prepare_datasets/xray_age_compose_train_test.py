import os
from parse import parse

absp = '/media/data10T_1/datasets/Voynov/xray_age'

ldir = os.listdir(absp)
f = '{0}_{1}'
splits = [
    # [[20, 30], [60, 70]],
    # [[20, 35], [55, 70]],
    # [[20, 40], [50, 70]],
    # [[20, 30], [40, 50], [60, 70]],
    # [[20, 30], [55, 70]],
    # [[20, 30], [50, 70]],
    # [[20, 35], [60, 70]],
    # [[20, 40], [60, 70]],
    # [[30, 44], [45, 60]]
]

import numpy
from sklearn.model_selection import train_test_split as tts

gend = ['m', 'f']

def dir_name_spl(spl):
    dir_f = '__xray_age_{0}'
    s = ''
    for r in spl:
        start = r[0]
        end = r[1]
        s += str(start) + '-' + str(end) + '_'
    dir_name = dir_f.format(s)
    return dir_name

for spl in splits:
    dir_name = dir_name_spl(spl) 
    train = 'train.txt'
    test = 'test.txt'
    
    if not os.path.exists(os.path.join('../', dir_name)):
        os.mkdir(os.path.join('../', dir_name))

    classes = range(len(spl))
    folders = []
    for cl in classes:
        start = spl[cl][0]
        end = spl[cl][1]
        class_folders = []
        for age in range(start, end + 1):
            for g in gend:
                class_folders += [os.path.join(absp, f.format(age, g))]
        folders += [class_folders]
    
    paths = []
    labels = []
    for cl in classes:
        class_folders = folders[cl]
        class_paths = []
        for folder in class_folders:
            folder_ldir = os.listdir(folder)
            folder_paths = [os.path.join(folder, d) for d in folder_ldir]
            class_paths += folder_paths
        paths += class_paths
        labels += [cl] * len(class_paths)
    paths = numpy.array(paths)
    labels = numpy.array(labels)
    
    indices = numpy.arange(0, len(labels))
    train_indices, test_indices, train_labels, test_labels = tts(indices, labels, stratify=labels, test_size=0.2, random_state=42)
    train_paths = paths[train_indices]
    test_paths = paths[test_indices]    

    train_file = open('../' + os.path.join(dir_name, train), 'w')
    lines = [t[0] + ' ' + str(t[1]) + '\n' for t in zip(train_paths, train_labels)]
    lines = lines[:len(lines) // 50 * 50]
    train_file.writelines(lines)
    train_file.close()
    
    test_file = open('../' + os.path.join(dir_name, test), 'w')
    lines = [t[0] + ' ' + str(t[1]) + '\n' for t in zip(test_paths, test_labels)]
    lines = lines[:len(lines) // 50 * 50]
    test_file.writelines(lines)
    test_file.close()



    



    
