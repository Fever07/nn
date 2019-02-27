import os
import numpy
from parse import parse
import sys
from matplotlib import pyplot as plt
import pickle
from utils_nn import parse_pred_file

def read_attack_results(absp, test=True):
    if test:
        f = 'a_test_PGD_eps_{0:0.2}_max_iter_{1}.pkl'
    else:
        f = 'a_train_PGD_eps_{0:0.2}_max_iter_{1}.pkl'

    iter_space = numpy.arange(1, 21)
    eps_space = numpy.arange(0.02, 0.22, 0.02)

    values = []
    for max_iter in iter_space:
        for max_eps in eps_space:
            filename = f.format(max_eps, max_iter)
            abs_filename = os.path.join(absp, filename)
            file = open(abs_filename, 'rb')
            a_probs = pickle.load(file)
            file.close()

            values.append(a_probs)

    values = numpy.array(values)
    shape = values.shape
    values = values.reshape(len(iter_space), len(eps_space), *shape[1: ])
    print(values.shape, values.dtype)
    return iter_space, eps_space, values

def read_pred_results(absp, test=True):
    if test:
        filename = 'pred_test.pkl'
    else:
        filename = 'pred_train.pkl'
    abs_filename = os.path.join(absp, filename)
    return parse_pred_file(abs_filename)

def rate_attacked(absp):
    def get_attacked_rate(a_probs):
        labels = numpy.argmax(probs, axis=1)
        a_labels = numpy.argmax(a_probs, axis=1)
        attacked = len(numpy.where((labels == a_labels) == False)[0])
        total = len(labels)
        return attacked / total

    iter_space, eps_space, a_probs = read_attack_results(absp, test=True)
    probs = read_pred_results(absp, test=True)

    # build a dependence of number of attacked images
    # on epsilon
    xs = numpy.repeat([eps_space], len(iter_space), axis=0)
    ys = []
    for a_iter_probs in a_probs:
        dependence_on_eps = list(map(get_attacked_rate, a_iter_probs))
        ys.append(dependence_on_eps)
    pts = list(zip(xs, ys))

    path = os.path.join(absp, 'test_attacked_rate.pkl')
    file = open(path, 'wb')
    pickle.dump(pts, file)
    file.close()

    return pts


def attacked_by_bins(absp):
    bins = numpy.arange(0.5, 1.05, 0.05)
    iter_space, eps_space, a_probs = read_attack_results(absp, test=True)
    probs = read_pred_results(absp, test=True)
    results = []
    for i, it in enumerate(iter_space):
        by_eps_results = []
        for j, eps in enumerate(eps_space):
            adv_probs = a_probs[i][j]
            orig_labels = numpy.argmax(probs, axis=1)
            adv_labels = numpy.argmax(adv_probs, axis=1)
            attacked_inds = numpy.where((orig_labels == adv_labels) == False)[0]
            attacked_total = len(attacked_inds)
            orig_class_probs = numpy.max(probs, axis=1)

            orig_bin_probs = numpy.digitize(orig_class_probs, bins)
            attacked_bin_probs = orig_bin_probs[attacked_inds]
            bin_inds = numpy.arange(0, len(bins)) + 1

            orig_in_bins = numpy.histogram(orig_bin_probs, bins=bin_inds)[0]
            attacked_in_bins = numpy.histogram(attacked_bin_probs, bins=bin_inds)[0]
            rate_in_bins = numpy.divide(attacked_in_bins, orig_in_bins)
            # replace nans by zero
            # as it is result of division by zero
            rate_in_bins[rate_in_bins != rate_in_bins] = 0
            
            by_eps_results.append(rate_in_bins)

        results.append(by_eps_results)

    file = open(os.path.join(absp, 'test_attacked_by_bins.pkl'), 'wb')
    pickle.dump(results, file)
    file.close()   

    return results 


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please provide a path to folder')
        exit()
    path_to_folder = sys.argv[1]
    rate_attacked(path_to_folder)
    attacked_by_bins(path_to_folder)
