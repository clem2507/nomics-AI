import os
import sys
import time
import argparse

from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/training/data_loader')
sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/training/learning_models')

from preprocessing import Preprocessing
from ann import train_model as ann_train_model
from knn import train_model as knn_train_model


def train(model, analysis_directory, segmentation_value, downsampling_value, epochs, data_balancing, neighbors, batch, standard_scale, sliding_window, stateful, center_of_interest, task):
    """
    Primary function for training the CNN, LSTM or KNN model

    Parameters:

    -model (--model): learning model architecture, either CNN, LSTM or KNN
    -analysis_directory (--analysis_directory): directory path containing the analyses to train the model on
    -segmentation_value (--segmentation_value): window segmentation value in minute
    -downsampling_value (--downsampling_value): signal downsampling value in second
    -epochs (--epochs): number of epochs to train the model
    -data_balancing (--data_balancing): true if balanced data is needed, false otherwise
    -neighbors (--neighbors): number of k-nearest neighbors to train the model for the KNN
    -batch (--batch): batch size for training
    -standard_scaled (--standard_scale): true to perform a standardizatin by centering and scaling the data
    -stateful (--stateful): true to use stateful LSTM instead of stateless
    -sliding_window (--sliding_window): true to use sliding window with a small center portion of interest
    -center_of_interest (--center_of_interest): center of interest size in seconds for the sliding window
    -task: (--task): corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep
    """

    segmentation_value = float(segmentation_value)
    downsampling_value = float(downsampling_value)

    analysis_directory = os.path.dirname(os.path.abspath('util.py')) + '/' + analysis_directory
    log_time = datetime.now().strftime('%d-%m-%Y %H-%M-%S')

    if os.path.exists(analysis_directory):
        Preprocessing(analysis_directory=analysis_directory, task=task).create_dataframes()
    else:
        raise Exception('input directory does not exist')
    
    if model.lower() == 'cnn' or model.lower() == 'lstm':
        ann_train_model(analysis_directory=analysis_directory, model=model.lower(), segmentation_value=segmentation_value, downsampling_value=downsampling_value, epochs=epochs, data_balancing=data_balancing, log_time=log_time, batch_size=batch, standard_scale=standard_scale, stateful=stateful, sliding_window=sliding_window, center_of_interest=center_of_interest, task=task)
    elif model.lower() == 'knn':
        knn_train_model(analysis_directory=analysis_directory, neighbors=neighbors, segmentation_value=segmentation_value, downsampling_value=downsampling_value, data_balancing=data_balancing, log_time=log_time, center_of_interest=center_of_interest, standard_scale=standard_scale)
    else:
        raise Exception('model type does not exist, please choose between LSTM, CNN and KNN')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM', help='learning model architecture - either CNN, LSTM or KNN')
    parser.add_argument('--analysis_directory', type=str, default='', help='directory path with analysis to use for neural network training')
    parser.add_argument('--segmentation_value', type=float, default=1, help='time series time split window in minutes')
    parser.add_argument('--downsampling_value', type=float, default=1, help='signal resampling in Hz')
    parser.add_argument('--epochs', type=int, default=5, help='total number of epochs for training')
    parser.add_argument('--no_balance', dest='data_balancing', action='store_false', help='invoke to not balance the dataset instances')
    parser.add_argument('--neighbors', type=int, default=3, help='neighbors: number of k-nearest neighbors to train the model for the KNN')
    parser.add_argument('--batch', type=int, default=30, help='batch size for training')
    parser.add_argument('--standard_scale', dest='standard_scale', action='store_true', help='invoke to perform a standardizatin by centering and scaling the data')
    parser.add_argument('--stateful', dest='stateful', action='store_true', help='invoke to use stateful LSTM instead of stateless')
    parser.add_argument('--sliding_window', dest='sliding_window', action='store_true', help='invoke to use sliding window with a small center portion of interest')
    parser.add_argument('--center_of_interest', type=int, default=10, help='center of interest size in seconds for the sliding window')
    parser.add_argument('--task', type=int, default='1', help='corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep')
    parser.set_defaults(data_balancing=True)
    parser.set_defaults(standard_scale=False)
    parser.set_defaults(stateful=False)
    parser.set_defaults(sliding_window=False)
    return parser.parse_args()


def main(p):
    train(**vars(p))


if __name__ == '__main__':
    # statement to avoid useless warnings during training
    # export TF_CPP_MIN_LOG_LEVEL=3
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    opt = parse_opt()
    main(p=opt)

    # CNN - valid/invalid
    # python3 code/training/train.py --analysis_directory 'data/valid_invalid_analysis' --segmentation_value 1 --downsampling_value 1 --epochs 200 --model 'cnn' --no_balance --batch 32 --task 1
    # CNN - awake/sleep
    # python3 code/training/train.py --analysis_directory 'data/awake_sleep_analysis' --segmentation_value 1 --downsampling_value 1 --epochs 200 --model 'cnn' --no_balance --batch 32 --task 2

    # LSTM - valid/invalid
    # python3 code/training/train.py --analysis_directory 'data/valid_invalid_analysis' --segmentation_value 1 --downsampling_value 1 --epochs 200 --model 'lstm' --no_balance --batch 32 --task 1
    # LSTM - awake/sleep
    # python3 code/training/train.py --analysis_directory 'data/awake_sleep_analysis' --downsampling_value 1 --epochs 200 --model 'lstm' --no_balance --batch 32 --stateful --task 2

    # KNN - not working anymore
