import os

from deep_learning_models.cnn_model import train_cnn
from deep_learning_models.lstm_model import train_lstm
from deep_learning_models.cnn_model import hyperparameters_tuning as ht_cnn
from deep_learning_models.lstm_model import hyperparameters_tuning as ht_lstm


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # train_cnn(time_split=5, time_resampling=3, epochs=5)
    # ht_cnn(time_split=1, time_resampling=3, max_trials=10, epochs=5, batch_size=32)
    # test_sequences(model_path=os.path.dirname(os.path.abspath('runscript.py')) + '/models/cnn/saved_model')

    train_lstm(time_split=5, time_resampling=3, epochs=5)
    # ht_lstm(time_split=1, time_resampling=3, max_trials=10, epochs=3, batch_size=32)
    # test_sequences(model_path=os.path.dirname(os.path.abspath('runscript.py')) + '/models/lstm/saved_model')
