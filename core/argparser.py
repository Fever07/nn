import argparse
from core.constants import MODELS

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('absp', type=str, help='Absolute path to a folder with dataset configuration')
    parser.add_argument('-m', '--model_name', default='inceptionv3', choices=MODELS.keys(), type=str, help='Name of neural network architecture')
    parser.add_argument('-n', '--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('-c', '--colored', default=False, type=bool, help='Set to true if images are colored, false otherwise')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')    
    parser.add_argument('--use_gpu', default=argparse.SUPPRESS, type=str, help='If specified, uses only selected gpu to run on')

    return vars(parser.parse_args())