import os

from training.learning_models.cnn_model import train_cnn
from training.learning_models.lstm_model import train_lstm
from training.learning_models.cnn_model import hyperparameters_tuning as ht_cnn
from training.learning_models.lstm_model import hyperparameters_tuning as ht_lstm

from training.data_loader.preprocessing import Preprocessing

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    num_class = 2
    epochs = 5
    baseline_model = True
    data_balancing = False

    train_cnn(time_split=1, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=1, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=1, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=1, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=1, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=1, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    #
    train_cnn(time_split=2, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=2, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=2, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=2, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=2, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=2, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    #
    train_cnn(time_split=3, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=3, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=3, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=3, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=3, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=3, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    #
    train_lstm(time_split=1, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=1, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=1, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=1, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=1, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=1, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)

    train_lstm(time_split=2, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=2, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=2, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=2, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=2, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=2, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    #
    train_lstm(time_split=3, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=3, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=3, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=3, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=3, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=3, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)

    # ht_cnn(time_split=1, time_resampling=1, max_trials=20, epochs=epochs, batch_size=32, num_class=num_class)
    #
    # ht_lstm(time_split=1, time_resampling=1, max_trials=20, epochs=epochs, batch_size=64, num_class=num_class)

    num_class = 3
    epochs = 50
    baseline_model = True
    data_balancing = False

    train_cnn(time_split=1, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=1, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=1, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=1, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=1, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=1, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    #
    train_cnn(time_split=2, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=2, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=2, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=2, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=2, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=2, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    #
    train_cnn(time_split=3, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=3, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=3, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=3, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_cnn(time_split=3, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_cnn(time_split=3, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    #
    train_lstm(time_split=1, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=1, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=1, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=1, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=1, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=1, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)

    train_lstm(time_split=2, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=2, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=2, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=2, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=2, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=2, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    #
    train_lstm(time_split=3, time_resampling=0.1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=3, time_resampling=0.2, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=3, time_resampling=0.5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=3, time_resampling=1, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    # train_lstm(time_split=3, time_resampling=3, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)
    train_lstm(time_split=3, time_resampling=5, epochs=epochs, num_class=num_class, baseline_model=baseline_model, data_balancing=data_balancing)

    # ht_cnn(time_split=1, time_resampling=1, max_trials=20, epochs=epochs, batch_size=32, num_class=num_class)
    #
    # ht_lstm(time_split=1, time_resampling=1, max_trials=20, epochs=epochs, batch_size=64, num_class=num_class)
