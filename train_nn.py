import os
import argparse

from data_loader.preprocessing import create_dataframes
from deep_learning_models.cnn_model import train_cnn
from deep_learning_models.lstm_model import train_lstm


def run(model, time_split, time_resampling, epochs, num_class):
    if epochs is None:
        if num_class == 2:
            epochs = 5
        else:
            epochs = 10

    if model.lower() == 'lstm':
        train_lstm(time_split=time_split, time_resampling=time_resampling, epochs=epochs, num_class=num_class)
    elif model.lower == 'cnn':
        train_cnn(time_split=time_split, time_resampling=time_resampling, epochs=epochs, num_class=num_class)
    else:
        raise Exception('model architecture type does not exist, please choose between LSTM and CNN')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM', help='deep learning model architecture - either CNN or LSTM')
    parser.add_argument('--time_split', type=int, default=3, help='time series time split window in minutes')
    parser.add_argument('--time_resampling', type=int, default=1, help='signal resampling in Hz')
    parser.add_argument('--epochs', type=int, default=None, help='total number of epochs for training')
    parser.add_argument('--num_class', type=int, default=3, help='number of classes for classification, input 2 for (valid | invalid), 3 for (valid | invalid | awake)')
    return parser.parse_args()


def main(p):
    run(**vars(p))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # cnn
    # binomial
    # python train_nn.py --model 'cnn' --time_split 1 --time_resampling 1 --epochs 5 --num_class 2
    # multinomial
    # python train_nn.py --model 'cnn' --time_split 3 --time_resampling 1 --epochs 10 --num_class 3

    # lstm
    # binomial
    # python train_nn.py --model 'lstm' --time_split 1 --time_resampling 1 --epochs 5 --num_class 2
    # multinomial
    # python train_nn.py --model 'lstm' --time_split 3 --time_resampling 1 --epochs 10 --num_class 3

    opt = parse_opt()
    main(p=opt)
