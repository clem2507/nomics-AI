import os
import time
import datetime
import argparse

from learning_models.cnn_model import train_cnn
from learning_models.lstm_model import train_lstm
from data_loader.preprocessing import create_dataframes


def train(model, analysis_directory, segmentation_value, downsampling_value, epochs, num_class, data_balancing):
    """
    Primary function for training the CNN or LSTM model

    Parameters:

    -model (--model): deep learning model architecture, either CNN or LSTM
    -analysis_directory (--analysis_directory): directory path containing the analyses to train the model on
    -segmentation_value (--segmentation_value): window segmentation value in minute
    -downsampling_value (--downsampling_value): signal downsampling value in second
    -epochs (--epochs): number of epochs to train the model
    -num_class (--num_class): number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -data_balancing (--data_balancing): true if balanced data is needed, false otherwise
    """

    segmentation_value = float(segmentation_value)
    downsampling_value = float(downsampling_value)

    analysis_directory = os.path.dirname(os.path.abspath('util.py')) + '/' + analysis_directory
    log_time = str(datetime.datetime.fromtimestamp(time.time()))

    print(analysis_directory)
    if os.path.exists(analysis_directory):
        create_dataframes(directory=analysis_directory)
    else:
        raise Exception('input directory does not exist')

    if epochs is None:
        if num_class == 2:
            epochs = 5
        else:
            epochs = 50
    if model.lower() == 'lstm':
        train_lstm(segmentation_value=segmentation_value, downsampling_value=downsampling_value, epochs=epochs, num_class=num_class, data_balancing=data_balancing, log_time=log_time)
    elif model.lower() == 'cnn':
        train_cnn(segmentation_value=segmentation_value, downsampling_value=downsampling_value, epochs=epochs, num_class=num_class, data_balancing=data_balancing, log_time=log_time)
    else:
        raise Exception('model architecture type does not exist, please choose between LSTM and CNN')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM', help='deep training model architecture - either CNN or LSTM')
    parser.add_argument('--analysis_directory', type=str, default='', help='directory path with analysis to use for neural network training')
    parser.add_argument('--segmentation_value', type=float, default=1, help='time series time split window in minutes')
    parser.add_argument('--downsampling_value', type=float, default=1, help='signal resampling in Hz')
    parser.add_argument('--epochs', type=int, default=None, help='total number of epochs for training')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes for classification, input 2 for (valid | invalid), 3 for (valid | invalid | awake)')
    parser.add_argument('--data_balancing', type=bool, default=False, help='dataset instances balancing, if false, no data balancing is performed among the different classes, otherwise balancing is applied')
    return parser.parse_args()


def main(p):
    train(**vars(p))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # CNN
    # binray
    # python3 training/train_nn.py --model 'cnn' --analysis_directory 'training/data/analysis' --segmentation_value 1 --downsampling_value 1 --epochs 5 --num_class 2 --data_balancing False
    # multinomia
    # python3 training/train_nn.py --model 'cnn' --analysis_directory 'training/data/analysis' --segmentation_value 1 --downsampling_value 1 --epochs 50 --num_class 3 --data_balancing False

    # LSTM
    # binary
    # python3 training/train_nn.py --model 'lstm' --analysis_directory 'training/data/analysis' --segmentation_value 1 --downsampling_value 1 --epochs 5 --num_class 2 --data_balancing False
    # multinomial
    # python3 training/train_nn.py --model 'lstm' --analysis_directory 'training/data/analysis' --segmentation_value 1 --downsampling_value 1 --epochs 50 --num_class 3 --data_balancing False

    opt = parse_opt()
    main(p=opt)
