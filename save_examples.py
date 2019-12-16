from matplotlib import pyplot as plt
import numpy
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K

from core.generator import ConfigurationGenerator
from core.utils_nn import parse_attack_files, \
parse_file, parse_pred_file, get_model_path, load_gray_image, load_color_image
from core.pgd_attack import PGD_attack as pgd

import shutil
from datetime import datetime
import os
from parse import parse
import csv

datasets = [
    {
        'absp': '../__histology',
        'label': 'Histology Camelyon',
        'color': 'C1',
        'input_shape': [256, 256, 3],
        'n_classes': 2,
        'colored': True
    },
    # {
    #     'absp': '../__aorta_razv',
    #     'label': 'Aorta',
    #     'color': 'C2',
    #     'input_shape': [256, 256, 1],
    #     'n_classes': 2,
    #     'colored': False,
    #     'a_bad': [2, 6, 7, 12, 16, 20, 21, 24, 32, 33, 34, 35, 38, 39, 40, 41, \
    #     42, 46, 52, 53, 54, 60, 61, 62, 63, 75, 77, 79, 82, 83, 84, 85, 86, 87, 88, 89, 90],
    #     'na_bad': [4, 5, 6, 7, 8, 9, 12, 15, 17, 18, 23, 24, 26, 27, 28, 29, 30, 32, 34, 40, \
    #     42, 46, 47, 49, 50, 53, 54, 57, 58, 60, 62, 63, 64, 66, 67, 69, 72, 74, 75, 76, 78, 80, \
    #     86, 87, 88, 91, 94, 95, 99]
    # },
    # {
    #     'absp': '../__xray',
    #     'label': 'Xray with noise',
    #     'color': 'C3',
    #     'input_shape': [256, 256, 1],
    #     'n_classes': 2,
    #     'colored': False
    # },
    # {
    #     'absp': '../__histology_tifs_4classes',
    #     'label': 'Histology 4cl',
    #     'color': 'C6',
    #     'input_shape': [256, 256, 3],
    #     'n_classes': 4,
    #     'colored': True,
    #     'a_bad': [1, 9, 10, 11, 13, 17, 35, 36, 41, 44, 48, 50, 52, 53, \
    #     62, 63, 64, 66, 72, 73, 74, 79, 80, 81, 84, 89, 94, 95],
    #     'na_bad': [0, 1, 6, 10, 33, 45, 50, 58, 62, 65, 67, 68, 69, 72, \
    #     73, 74, 75, 79, 82, 83, 97, 98, 99]
    # },
    {
        'absp': '../__histology_tifs_2classes_thyroid',
        'label': 'Histology thyr',
        'color': '',
        'input_shape': [256, 256, 3],
        'n_classes': 2,
        'colored': True
    },
    {
        'absp': '../__histology_tifs_2classes_ovary',
        'label': 'Histology thyr',
        'color': '',
        'input_shape': [256, 256, 3],
        'n_classes': 2,
        'colored': True
    }
    # {
    #     'absp': '../__xray_age_new_20-35_50-70_',
    #     'input_shape': [256, 256, 1],
    #     'n_classes': 2,
    #     'colored': False,
    #     'label': 'X-N'
    # }
]

test_file = 'test.txt'
pred_test_file = 'pred_test.pkl'
a_test_f = 'a_test_PGD_eps_{0:0.2}_max_iter_{1}.pkl'
examples_folder = 'attack_examples'
iter_space = numpy.arange(1, 21)
eps_space = numpy.arange(0.02, 0.22, 0.02)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sess = K.get_session()

rng = numpy.random.RandomState(1337)
for dataset in datasets:
    absp = dataset['absp']
    paths, labels = parse_file(os.path.join(absp, test_file))
    labels = numpy.array(labels)
    probs = parse_pred_file(os.path.join(absp, pred_test_file))
    a_probs = parse_attack_files(os.path.join(absp, a_test_f),
                                 iter_space=iter_space,
                                 eps_space=eps_space)
    pred_labels = numpy.argmax(probs, axis=1)
    a_mineps_probs = a_probs[-1, 0, :, :]
    a_maxeps_probs = a_probs[-1, -1, :, :]
    a_min_labels = numpy.argmax(a_mineps_probs, axis=1)
    a_max_labels = numpy.argmax(a_maxeps_probs, axis=1)
    a_inds_min = numpy.where((pred_labels - a_min_labels) != 0)[0]
    a_inds_max = numpy.where((pred_labels - a_max_labels) != 0)[0]
    a_inds = set(a_inds_min).intersection(a_inds_max)
    a_inds = list(a_inds)
    not_a_inds = set(list(range(len(paths)))).difference(a_inds)
    not_a_inds = list(not_a_inds)
    
    paths = numpy.array(paths)

    a_paths = paths[a_inds]
    input_shape = dataset['input_shape']
    n_classes = dataset['n_classes']
    colored = dataset['colored']
    
    print(len(a_inds), len(not_a_inds))
    
    abs_ex_f = os.path.join(absp, examples_folder)
    if not os.path.exists(abs_ex_f):
        os.mkdir(abs_ex_f)
        os.mkdir(os.path.join(abs_ex_f, 'attacked'))
        os.mkdir(os.path.join(abs_ex_f, 'not_attacked'))
    n = 200
    a_inds = a_inds[:n]
    not_a_inds = not_a_inds[:n]
    a_imgs = []
    not_a_imgs = []
    
    batch_size = 20
    
    model_path = get_model_path(absp)
    model = load_model(model_path)
    features_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    print(dataset['label'])

    mode = 'RGB' if colored else 'L'
    load_image = load_color_image if colored else load_gray_image
    f = '{}-{}.png'
    for i, a_ind in enumerate(a_inds):
        img = load_image(paths[a_ind])
        img = Image.fromarray(numpy.squeeze((img * 255.0).astype(numpy.uint8)), mode=mode) 
        img.save(os.path.join(abs_ex_f, 'attacked', f.format(i, labels[a_ind])))
    # for i, a_ind in enumerate(not_a_inds):
    #     img = load_image(paths[a_ind])
    #     img = Image.fromarray(numpy.squeeze((img * 255.0).astype(numpy.uint8)), mode=mode) 
    #     img.save(os.path.join(abs_ex_f, 'not_attacked', f.format(i, labels[a_ind])))
        
    # features = numpy.empty(shape=[n, 2049], dtype=numpy.float32)
    # a_features = numpy.empty(shape=[n, 2049], dtype=numpy.float32)
    # pgd_attacker = pgd(model,
    #                        max_epsilon=0.1,
    #                        max_iter=iter_space[-1],
    #                        batch_shape=[None]+input_shape,
    #                        initial_lr=1,
    #                        lr_decay=1,
    #                        targeted=False,
    #                        n_classes=n_classes,
    #                        img_bounds=[0, 1],
    #                        rng=rng)
    # for it in range(int(n // batch_size)):
    #     print('Processing {0}/{1} images - {2}'.format(
    #         (it + 1) * batch_size,
    #         n,
    #         datetime.now().time().strftime('%H:%M:%S')
    #     ))
    #     it_imgs = []
    #     it_labels = []
    #     it_start = it * batch_size
    #     it_end = (it + 1) * batch_size
    #     it_a_inds = a_inds[it_start: it_end]
    #     for a_ind in it_a_inds:
    #         it_imgs.append(load_image(paths[a_ind]))
    #         it_labels.append(a_ind)
    #     it_imgs = numpy.array(it_imgs)
    #     it_labels = numpy.array(it_labels)
    #     features[it_start: it_end, 1:] = features_model.predict(it_imgs)

    #     it_a_imgs = pgd_attacker.generate(sess, it_imgs, it_labels)
    #     a_features[it_start: it_end, 1:] = features_model.predict(it_a_imgs)
    # features[:, 0] = labels[a_inds]
    # a_features[:, 0] = 1 - labels[a_inds]
    # with open(os.path.join(abs_ex_f, 'attacked', 'features.csv'), 'w', newline='') as file:
    #     wr = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for feat in features:
    #         wr.writerow(feat.tolist())

    # with open(os.path.join(abs_ex_f, 'attacked', 'a_features.csv'), 'w', newline='') as file:
    #     wr = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for feat in a_features:
    #         wr.writerow(feat.tolist())


    # for max_eps in eps_space:
    for max_eps in [0.02]:
        if not os.path.exists(os.path.join(abs_ex_f, 'attacked', 'eps-{0:0.2}'.format(max_eps))):
            os.mkdir(os.path.join(abs_ex_f, 'attacked', 'eps-{0:0.2}'.format(max_eps)))
        pgd_attacker = pgd(model,
                           max_epsilon=max_eps,
                           max_iter=iter_space[-1],
                           batch_shape=[None]+input_shape,
                           initial_lr=1,
                           lr_decay=1,
                           targeted=False,
                           n_classes=n_classes,
                           img_bounds=[0, 1],
                           rng=rng)
        
#         a_probs = numpy.empty(shape=[max_iter, 0, n_classes], dtype=numpy.float32)
        imgs = numpy.array([load_image(path) for path in paths[a_inds]])
        img_labels = labels[a_inds]
        for i in range(n // batch_size):            
            a_imgs_batch = imgs[i * batch_size: (i + 1) *batch_size]
            labels_batch = img_labels[i * batch_size: (i + 1) *batch_size]
            print('Processing {0}/{1} images - {2}'.format(
                (i + 1) * batch_size,
                n,
                datetime.now().time().strftime('%H:%M:%S')
            ))

            # Attack "imgs": do "max_iter" iterations, and receive
            # generated images and probs for each iteration
            a_imgs_for_iters, a_probs_for_iters = pgd_attacker.generate_imgs_and_probs(sess,
                                                                                       a_imgs_batch,
                                                                                       labels_batch)

#             a_probs_for_iters = numpy.array(a_probs_for_iters)
            # for j, it in enumerate(iter_space):
            for j, it in [[-1, 20]]:
                a_imgs_for_iter = a_imgs_for_iters[j]
                
                # f = '{0}-{1}_eps_{2:0.2}_iter_{3}.png'
                # f = os.path.join(os.path.join(abs_ex_f, 'attacked', 'eps-{0:0.2}'.format(max_eps), f))

                f = '{0}-{1}_attack.png'
                f = os.path.join(abs_ex_f, 'attacked', f)

                a_imgs = []
                for a_img in a_imgs_for_iter:
                    assert 0 <= a_img.min() and a_img.max() <= 1

                a_imgs = a_imgs_for_iter
                a_imgs = numpy.array(a_imgs)
                a_imgs = [Image.fromarray(numpy.squeeze(mg), mode=mode) for mg in (a_imgs * 255.0).astype(numpy.uint8)]
                for k, img in enumerate(a_imgs):
                    img.save(f.format(i * batch_size + k, labels_batch[k]))

#     for max_eps in eps_space:
#         if not os.path.exists(os.path.join(abs_ex_f, 'not_attacked', 'eps-{0:0.2}'.format(max_eps))):
#             os.mkdir(os.path.join(abs_ex_f, 'not_attacked', 'eps-{0:0.2}'.format(max_eps)))
#         pgd_attacker = pgd(model,
#                            max_epsilon=max_eps,
#                            max_iter=iter_space[-1],
#                            batch_shape=[None]+input_shape,
#                            initial_lr=1,
#                            lr_decay=1,
#                            targeted=False,
#                            n_classes=n_classes,
#                            img_bounds=[0, 1],
#                            rng=rng)
        
# #         a_probs = numpy.empty(shape=[max_iter, 0, n_classes], dtype=numpy.float32)
#         imgs = numpy.array([load_image(path) for path in paths[not_a_inds]])
#         img_labels = labels[not_a_inds]
#         for i in range(n // batch_size):            
#             a_imgs_batch = imgs[i * batch_size: (i + 1) *batch_size]
#             labels_batch = img_labels[i * batch_size: (i + 1) *batch_size]
#             print('Processing {0}/{1} images - {2}'.format(
#                 (i + 1) * batch_size,
#                 n,
#                 datetime.now().time().strftime('%H:%M:%S')
#             ))

#             # Attack "imgs": do "max_iter" iterations, and receive
#             # generated images and probs for each iteration
#             a_imgs_for_iters, a_probs_for_iters = pgd_attacker.generate_imgs_and_probs(sess,
#                                                                                        a_imgs_batch,
#                                                                                        labels_batch)
# #             a_probs_for_iters = numpy.array(a_probs_for_iters)
#             for j, it in enumerate(iter_space):
#                 a_imgs_for_iter = a_imgs_for_iters[j]
                
#                 f = '{0}-{1}_eps_{2:0.2}_iter_{3}.png'
#                 f = os.path.join(os.path.join(abs_ex_f, 'not_attacked', 'eps-{0:0.2}'.format(max_eps), f))

#                 a_imgs = []
#                 for a_img in a_imgs_for_iter:
#                     assert 0 <= a_img.min() and a_img.max() <= 1

#                 a_imgs = a_imgs_for_iter
#                 a_imgs = numpy.array(a_imgs)
#                 a_imgs = [Image.fromarray(numpy.squeeze(mg), mode=mode) for mg in (a_imgs * 255.0).astype(numpy.uint8)]
#                 for k, img in enumerate(a_imgs):
#                     img.save(f.format(i * batch_size + k, labels_batch[k], max_eps, it))
        

# for dataset in datasets:
#     cwd = os.getcwd()
#     absp = dataset['absp']
#     os.chdir(absp)

#     a_inds = list(range(100))
#     na_inds = list(range(100))
#     if 'a_bad' in dataset:
#         a_inds = list(set(a_inds).difference(set(dataset['a_bad'])))
#     if 'na_bad' in dataset:
#         na_inds = list(set(na_inds).difference(set(dataset['na_bad'])))
#     a_inds.sort()
#     na_inds.sort()
#     a_inds = a_inds[:30]
#     na_inds = na_inds[:30]
#     assert len(a_inds) == 30 and len(na_inds) == 30

#     if not os.path.exists('show_ex'):
#         os.mkdir('show_ex')

#     if not os.path.exists('show_ex/not_attacked'):
#         os.mkdir('show_ex/not_attacked')
#         os.mkdir('show_ex/not_attacked/min_ampl-max_ampl')
#         os.mkdir('show_ex/not_attacked/ampl')
#         os.mkdir('show_ex/not_attacked/iter')

#     ldir = os.listdir('attack_examples/not_attacked')

#     filenames = [l for l in ldir if l.endswith('.png')]
#     imgnames = []
#     for i, num in enumerate(na_inds):
#         filename = [f for f in filenames if f.startswith(str(num) + '-')][0]
#         f = '{}.png'
#         base = parse(f, filename)[0]
#         shutil.copy('attack_examples/not_attacked/' + filename, 'show_ex/not_attacked/' + filename)

#         if i < 20:
#             shutil.copy('attack_examples/not_attacked/' + filename, 'show_ex/not_attacked/min_ampl-max_ampl/' + filename)

#             shutil.copy('attack_examples/not_attacked/eps-0.02/' + base + '_eps_0.02_iter_20.png', \
#                 'show_ex/not_attacked/min_ampl-max_ampl/' + base + '_eps_0.02_iter_20.png')
#             shutil.copy('attack_examples/not_attacked/eps-0.2/' + base + '_eps_0.2_iter_20.png', \
#                 'show_ex/not_attacked/min_ampl-max_ampl/' + base + '_eps_0.20_iter_20.png')

#         elif i < 25:
#             shutil.copy('attack_examples/not_attacked/' + filename, 'show_ex/not_attacked/iter/' + filename)

#             for it in range(1, 21):
#                 shutil.copy('attack_examples/not_attacked/eps-0.2/' + base + '_eps_0.2_iter_{}.png'.format(it), \
#                     'show_ex/not_attacked/iter/' + base + '_eps_0.20_iter_{}.png'.format(it))

#         else:
#             shutil.copy('attack_examples/not_attacked/' + filename, 'show_ex/not_attacked/ampl/' + filename)

#             for eps in numpy.arange(0.02, 0.22, 0.02):
#                 shutil.copy('attack_examples/not_attacked/eps-{0:0.2}/'.format(eps) + base + '_eps_{0:0.2}_iter_20.png'.format(eps), \
#                     'show_ex/not_attacked/ampl/' + base + '_eps_{:04.2F}_iter_20.png'.format(eps))

# for dataset in datasets:
#     cwd = os.getcwd()
#     absp = dataset['absp']
#     os.chdir(absp)

#     a_inds = list(range(100))
#     na_inds = list(range(100))
#     if 'a_bad' in dataset:
#         a_inds = list(set(a_inds).difference(set(dataset['a_bad'])))
#     if 'na_bad' in dataset:
#         na_inds = list(set(na_inds).difference(set(dataset['na_bad'])))
#     a_inds.sort()
#     na_inds.sort()
#     a_inds = a_inds[:30]
#     na_inds = na_inds[:30]
#     assert len(a_inds) == 30 and len(na_inds) == 30

#     if not os.path.exists('show_ex'):
#         os.mkdir('show_ex')

#     if not os.path.exists('show_ex/attacked'):
#         os.mkdir('show_ex/attacked')
#         os.mkdir('show_ex/attacked/min_ampl-max_ampl')
#         os.mkdir('show_ex/attacked/ampl')
#         os.mkdir('show_ex/attacked/iter')

#     ldir = os.listdir('attack_examples/attacked')

#     filenames = [l for l in ldir if l.endswith('.png')]
#     imgnames = []
#     for i, num in enumerate(a_inds):
#         filename = [f for f in filenames if f.startswith(str(num) + '-')][0]
#         f = '{}.png'
#         base = parse(f, filename)[0]
#         shutil.copy('attack_examples/attacked/' + filename, 'show_ex/attacked/' + filename)

#         if i < 20:
#             shutil.copy('attack_examples/attacked/' + filename, 'show_ex/attacked/min_ampl-max_ampl/' + filename)

#             shutil.copy('attack_examples/attacked/eps-0.02/' + base + '_eps_0.02_iter_20.png', \
#                 'show_ex/attacked/min_ampl-max_ampl/' + base + '_eps_0.02_iter_20.png')
#             shutil.copy('attack_examples/attacked/eps-0.2/' + base + '_eps_0.2_iter_20.png', \
#                 'show_ex/attacked/min_ampl-max_ampl/' + base + '_eps_0.20_iter_20.png')

#         elif i < 25:
#             shutil.copy('attack_examples/attacked/' + filename, 'show_ex/attacked/iter/' + filename)

#             for it in range(1, 21):
#                 shutil.copy('attack_examples/attacked/eps-0.2/' + base + '_eps_0.2_iter_{}.png'.format(it), \
#                     'show_ex/attacked/iter/' + base + '_eps_0.20_iter_{}.png'.format(it))

#         else:
#             shutil.copy('attack_examples/attacked/' + filename, 'show_ex/attacked/ampl/' + filename)

#             for eps in numpy.arange(0.02, 0.22, 0.02):
#                 shutil.copy('attack_examples/attacked/eps-{0:0.2}/'.format(eps) + base + '_eps_{0:0.2}_iter_20.png'.format(eps), \
#                     'show_ex/attacked/ampl/' + base + '_eps_{:04.2F}_iter_20.png'.format(eps))



        

    
    
    
    