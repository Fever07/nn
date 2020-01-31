import os
import sys
from sklearn.metrics import roc_curve, auc
from core.utils import parse_file, parse_pred_file
from matplotlib import pyplot as plt
import numpy

def save_roc_curve():
    datasets = [
        {
            'absp': '../__histology',
            'label': 'H-Mt'
        },
        {
            'absp': '../__aorta_razv',
            'label': 'X-Ao'
        },
        {
            'absp': '../__xray',
            'label': 'X-Lung'
        },
        {
            'absp': '../__histology_tifs_2classes_ovary',
            'label': 'H-Ov'
        },
        {
            'absp': '../__histology_tifs_2classes_thyroid',
            'label': 'H-Th'
        },
        # {
        #     'absp': '../__histology_tifs_4classes',
        #     'label': 'H-Ov-Th'
        # },
    ]

    def get_fpr_tpr_auc(dataset):
        abs_testp = os.path.join(dataset['absp'], 'test.txt')
        abs_pred_testp = os.path.join(dataset['absp'], 'pred_test.pkl')
        paths, labels = parse_file(abs_testp)
        probs = parse_pred_file(abs_pred_testp)
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
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(os.path.join('plots', 'roc_curve.png'))
    # plt.show()    

if __name__ == '__main__':
    save_roc_curve()
