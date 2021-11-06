import os

from deep_learning_models.cnn_model import train_cnn
from deep_learning_models.lstm_model import train_lstm
from deep_learning_models.cnn_model import hyperparameters_tuning as ht_cnn
from deep_learning_models.lstm_model import hyperparameters_tuning as ht_lstm

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    num_class = 3

    if num_class == 2:
        epochs = 5
    else:
        epochs = 10

    # train_cnn(time_split=1, time_resampling=0.1, epochs=epochs, num_class=num_class)
    # train_cnn(time_split=1, time_resampling=1, epochs=epochs, num_class=num_class)
    # train_cnn(time_split=1, time_resampling=3, epochs=epochs, num_class=num_class)

    # train_cnn(time_split=3, time_resampling=0.1, epochs=epochs, num_class=num_class)
    # train_cnn(time_split=3, time_resampling=1, epochs=epochs, num_class=num_class)
    # train_cnn(time_split=3, time_resampling=3, epochs=epochs, num_class=num_class)

    # train_cnn(time_split=5, time_resampling=1, epochs=epochs, num_class=num_class)

    # train_lstm(time_split=1, time_resampling=0.1, epochs=epochs, num_class=num_class)
    # train_lstm(time_split=1, time_resampling=1, epochs=epochs, num_class=num_class)
    # train_lstm(time_split=1, time_resampling=3, epochs=epochs, num_class=num_class)

    # train_lstm(time_split=3, time_resampling=0.1, epochs=epochs, num_class=num_class)
    # train_lstm(time_split=3, time_resampling=1, epochs=epochs, num_class=num_class)
    # train_lstm(time_split=3, time_resampling=3, epochs=epochs, num_class=num_class)

    # train_lstm(time_split=5, time_resampling=1, epochs=epochs, num_class=num_class)

    # ht_cnn(time_split=1, time_resampling=3, max_trials=10, epochs=epochs, batch_size=32, num_class=num_class)
    # test_sequences(model_path=os.path.dirname(os.path.abspath('run_script.py')) + '/models/cnn/saved_model')

    # ht_lstm(time_split=1, time_resampling=3, max_trials=10, epochs=epochs, batch_size=32, , num_class=num_class)
    # test_sequences(model_path=os.path.dirname(os.path.abspath('run_script.py')) + '/models/lstm/saved_model')





