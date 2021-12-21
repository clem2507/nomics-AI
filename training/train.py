import os
import sys
import time
import datetime
import argparse

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/training/data_loader')
sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/training/learning_models')

from preprocessing import Preprocessing
from ann import train_model as ann_train_model
from knn import train_model as knn_train_model


def train(model, analysis_directory, segmentation_value, downsampling_value, epochs, num_class, data_balancing, neighbors, stateful):
    """
    Primary function for training the CNN, LSTM or KNN model

    Parameters:

    -model (--model): learning model architecture, either CNN, LSTM or KNN
    -analysis_directory (--analysis_directory): directory path containing the analyses to train the model on
    -segmentation_value (--segmentation_value): window segmentation value in minute
    -downsampling_value (--downsampling_value): signal downsampling value in second
    -epochs (--epochs): number of epochs to train the model
    -num_class (--num_class): number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -data_balancing (--data_balancing): true if balanced data is needed, false otherwise
    -neighbors (--neighbors): number of k-nearest neighbors to train the model for the KNN
    -stateful (--stateful): true to use stateful parameter for LSTM training, false to use stateless
    """

    segmentation_value = float(segmentation_value)
    downsampling_value = float(downsampling_value)

    analysis_directory = os.path.dirname(os.path.abspath('util.py')) + '/' + analysis_directory
    log_time = str(datetime.datetime.fromtimestamp(time.time()))

    print(analysis_directory)
    if os.path.exists(analysis_directory):
        Preprocessing(analysis_directory=analysis_directory).create_dataframes()
    else:
        raise Exception('input directory does not exist')

    if epochs is None:
        if num_class == 2:
            epochs = 5
        else:
            epochs = 50
    
    if model.lower() == 'cnn' or model.lower() == 'lstm':
        ann_train_model(analysis_directory=analysis_directory, model=model.lower(), segmentation_value=segmentation_value, downsampling_value=downsampling_value, epochs=epochs, num_class=num_class, data_balancing=data_balancing, log_time=log_time, stateful=stateful)
    elif model.lower() == 'knn':
        knn_train_model(analysis_directory=analysis_directory, neighbors=neighbors, segmentation_value=segmentation_value, downsampling_value=downsampling_value, num_class=num_class, data_balancing=data_balancing, log_time=log_time)
    else:
        raise Exception('model type does not exist, please choose between LSTM, CNN and KNN')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM', help='learning model architecture - either CNN, LSTM or KNN')
    parser.add_argument('--analysis_directory', type=str, default='', help='directory path with analysis to use for neural network training')
    parser.add_argument('--segmentation_value', type=float, default=1, help='time series time split window in minutes')
    parser.add_argument('--downsampling_value', type=float, default=1, help='signal resampling in Hz')
    parser.add_argument('--epochs', type=int, default=None, help='total number of epochs for training')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes for classification, input 2 for (valid | invalid), 3 for (valid | invalid | awake)')
    parser.add_argument('--no_balance', dest='data_balancing', action='store_false', help='invoke to not balance the dataset instances')
    parser.add_argument('--neighbors', type=int, default=3, help='neighbors: number of k-nearest neighbors to train the model for the KNN')
    parser.add_argument('--stateful', dest='stateful', action='store_true', help='invoke to use stateful for LSTM training')
    parser.set_defaults(data_balancing=True)
    parser.set_defaults(stateful=False)
    return parser.parse_args()


def main(p):
    train(**vars(p))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    opt = parse_opt()
    main(p=opt)

    # CNN
    # binary
    # python3 training/train.py --analysis_directory 'training/data/analysis' --segmentation_value 1 --downsampling_value 1 --epochs 20 --num_class 2 --model 'cnn' --no_balance
    # multinomial
    # python3 training/train.py --analysis_directory 'training/data/analysis' --segmentation_value 15 --downsampling_value 0.1 --epochs 50 --num_class 3 --model 'cnn' --no_balance

    # LSTM
    # binary
    # python3 training/train.py --analysis_directory 'training/data/analysis' --segmentation_value 1 --downsampling_value 1 --epochs 20 --num_class 2 --model 'lstm' --no_balance --stateful
    # multinomial
    # python3 training/train.py --analysis_directory 'training/data/analysis' --segmentation_value 15 --downsampling_value 1 --epochs 50 --num_class 3 --model 'lstm' --no_balance --stateful

    # KNN
    # binary
    # python3 training/train.py --analysis_directory 'training/data/analysis' --segmentation_value 1 --downsampling_value 1 --num_class 2 --neighbors 5 --model 'knn' --no_balance
    # multinomial
    # python3 training/train.py --analysis_directory 'training/data/analysis' --segmentation_value 15 --downsampling_value 1 --num_class 3 --neighbors 5 --model 'knn' --no_balance