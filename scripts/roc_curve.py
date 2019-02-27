import os
import sys
from sklearn.metrics import roc_curve, auc
from utils_nn import parse_file
from matplotlib import pyplot as plt
import numpy

def save_roc_curve():
    datasets = [
        {
            'absp': '../__histology',
            'label': 'Гистология'
        },
        {
            'absp': '../__aorta_razv',
            'label': 'Аорта'
        },
        {
            'absp': '../__xray',
            'label': 'Рентген Легких'
        }
    ]

    def get_fpr_tpr_auc(dataset):
        abs_testp = os.path.join(dataset['absp'], 'test.txt')
        abs_pred_testp = os.path.join(dataset['absp'], 'pred_test.txt')
        paths, labels = parse_file(abs_testp)
        paths, probs = parse_file(abs_pred_testp, file_type='pred')
        probs = numpy.array(probs)

        fpr, tpr, thr = roc_curve(numpy.array(labels), probs.T[1]) 
        auc_ = auc(fpr, tpr)
        return fpr, tpr, auc_

    fprs = []
    tprs = []
    aucs = []

    for dataset in datasets:
        fpr_, tpr_, auc_ = get_fpr_tpr_auc(dataset)
        fprs += [fpr_]
        tprs += [tpr_]
        aucs += [auc_]

    plt.figure()
    for i, ds in enumerate(datasets):
        plt.plot(fprs[i], tprs[i], '-', label=ds['label'] + ', AUC = {0:0.3}'.format(aucs[i]))
    plt.plot([0, 1], [0, 1], '--', color='black')
    plt.xlabel('Ложно-положительная доля')
    plt.ylabel('Истинно-положительная доля')
    plt.legend()
    plt.savefig(os.path.join('../', 'roc_curve.png'))
    plt.show()    

if __name__ == '__main__':
    save_roc_curve()
