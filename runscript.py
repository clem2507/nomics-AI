from deep_learning_models.cnn_model import train_CNN
from deep_learning_models.lstm_model import train_LSTM
from deep_learning_models.cnn_model import hyperparameters_tuning as ht_cnn
from deep_learning_models.lstm_model import hyperparameters_tuning as ht_lstm

from utils.util import f1_m, sequence_test

if __name__ == '__main__':

    # train_CNN(load_flag=False, create_flag=True)
    # ht_lstm()
    # sequence_test(model_path='models/cnn/saved_model')

    train_LSTM(load_flag=False, create_flag=True)
    # ht_cnn()
    # sequence_test(model_path='models/lstm/saved_model')
