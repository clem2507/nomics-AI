import os
import argparse

from datetime import datetime
from data_loader import DataLoader
from ann import train_model


def train(model, 
          analysis_directory, 
          segmentation_value, 
          downsampling_value, 
          epochs, 
          data_balancing, 
          batch, 
          patience, 
          sliding_window, 
          center_of_interest, 
          task, 
          wandb):
    """
    Primary function for training the MLP, FCN, ResNet or LSTM model

    Parameters:

    -model (--model): learning model architecture, either MLP, FCN, ResNet or LSTM
    -analysis_directory (--analysis_directory): directory path containing the analyses to train the model on
    -segmentation_value (--segmentation_value): window segmentation value in second
    -downsampling_value (--downsampling_value): signal downsampling value in second
    -epochs (--epochs): number of epochs to train the model
    -data_balancing (--data_balancing): true if balanced data is needed, false otherwise
    -batch (--batch): batch size for training
    -patience (--patience): number of epochs without learning before early stop the training
    -sliding_window (--sliding_window): true to use sliding window with a small center portion of interest
    -center_of_interest (--center_of_interest): center of interest size in seconds for the sliding window
    -task (--task): corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep
    -wandb (--wandb): true to log the training results on weights and biases
    """

    segmentation_value = float(segmentation_value)
    downsampling_value = float(downsampling_value)

    analysis_directory = os.path.dirname(os.path.abspath('util.py')) + '/' + analysis_directory
    log_time = datetime.now().strftime('%d-%m-%Y %H-%M-%S')

    if os.path.exists(analysis_directory):
        DataLoader(analysis_directory=analysis_directory, task=task).create_dataframes()
    else:
        raise Exception('input directory does not exist')
    
    if model.lower() == 'fcn' or model.lower() == 'lstm' or model.lower() == 'resnet' or model.lower() == 'mlp':
        train_model(analysis_directory=analysis_directory, 
                    model_name=model.lower(), 
                    segmentation_value=segmentation_value, 
                    downsampling_value=downsampling_value, 
                    epochs=epochs, 
                    data_balancing=data_balancing, 
                    log_time=log_time, 
                    batch_size=batch, 
                    patience=patience, 
                    sliding_window=sliding_window, 
                    center_of_interest=center_of_interest, 
                    task=task, 
                    wandb_log=wandb)
    else:
        raise Exception('model type does not exist, please choose between MLP, FCN, ResNet or LSTM')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', help='learning model architecture - either MLP, FCN, ResNet or LSTM')
    parser.add_argument('--analysis_directory', type=str, default='', help='directory path with analysis to use for neural network training')
    parser.add_argument('--segmentation_value', type=float, default=300, help='time series time split window in seconds')
    parser.add_argument('--downsampling_value', type=float, default=1, help='signal resampling in Hz')
    parser.add_argument('--epochs', type=int, default=5, help='total number of epochs for training')
    parser.add_argument('--balance', dest='data_balancing', action='store_true', help='invoke to not balance the dataset instances')
    parser.add_argument('--batch', type=int, default=32, help='batch size for training')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs without learning before early stop the training')
    parser.add_argument('--sliding_window', dest='sliding_window', action='store_true', help='invoke to use sliding window with a small center portion of interest')
    parser.add_argument('--center_of_interest', type=int, default=60, help='center of interest size in seconds for the sliding window')
    parser.add_argument('--task', type=int, default='1', help='corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep')
    parser.add_argument('--wandb', dest='wandb', action='store_true', help='invoke to log the training results on weights and biases')
    parser.set_defaults(data_balancing=False)
    parser.set_defaults(sliding_window=False)
    parser.set_defaults(wandb=False)
    return parser.parse_args()


def main(p):
    train(**vars(p))


if __name__ == '__main__':
    # statement to avoid useless warnings during training
    # export TF_CPP_MIN_LOG_LEVEL=3
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    opt = parse_opt()
    main(p=opt)
