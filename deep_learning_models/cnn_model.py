import os
import time
import numpy as np
import keras_tuner as kt

from keras_tuner.tuners import RandomSearch
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import save_model
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from data_loader.preprocessing import Preprocessing, load_data
from utils.util import plot_confusion_matrix, f1_m


def evaluate_model(time_split, time_interval):
    X_train, y_train, X_test, y_test = Preprocessing(time_split=time_split, time_interval=time_interval).create_dataset()
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    X_train, y_train = shuffle(X_train, y_train)
    validation_split, verbose, epochs, batch_size = 0.1, 2, 3, 32
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # sigmoid activation function better than softmax for binary classification
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
    print(model.summary())
    # fit network
    model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy, *r = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)

    predictions = model.predict(X_test)
    classes = np.argmax(predictions, axis=1)

    int_y_test = []
    for item in y_test:
        int_y_test.append(np.argmax(item))

    cm = confusion_matrix(y_true=int_y_test, y_pred=classes)

    plot_confusion_matrix(cm=cm, classes=['Invalid signal', 'Valid signal'], title='Confusion Matrix')

    # Save the model
    os.mkdir(os.path.dirname(os.path.abspath('runscript.py')) + f'/models/cnn/split_{time_split}_resampling_{time_interval}')
    filepath = os.path.dirname(os.path.abspath('runscript.py')) + f'/models/cnn/split_{time_split}_resampling_{time_interval}/saved_model'
    model_info_file = open(os.path.dirname(os.path.abspath('runscript.py')) + f'/models/cnn/split_{time_split}_resampling_{time_interval}/info.txt', 'w')
    model_info_file.write(f'This file contains information about the CNN model accuracy using a {time_split} minutes signal time split and {time_interval} minutes median data resampling \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'num of epochs = {epochs} \n')
    model_info_file.write(f'last epoch test accuracy = {accuracy} \n')
    model_info_file.write(f'confusion matrix (invalid | valid) = \n {cm} \n')
    model_info_file.write('--- \n')
    model_info_file.close()
    save_model(model, filepath)
    return accuracy


def hyperparameters_tuning(time_split, time_interval):

    def build_model(hp):
        n_timesteps, n_features, n_outputs = 1800, 1, 2
        model = Sequential()
        model.add(Conv1D(filters=hp.Int('input_number_filters', min_value=32, max_value=160, step=32), kernel_size=hp.Int('input_kernel size', min_value=2, max_value=5, step=1), activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Conv1D(filters=hp.Int('conv_layer_1_number_filters', min_value=32, max_value=160, step=32), kernel_size=hp.Int('conv_layer_1_kernel size', min_value=2, max_value=5, step=1), activation='relu'))
        model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(hp.Int('dense_layer_units', min_value=32, max_value=544, step=64), activation='relu'))
        model.add(Dense(n_outputs, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    LOG_DIR = f'{int(time.time())}'
    X_train, y_train, X_test, y_test = Preprocessing(time_split=time_split, time_interval=time_interval).create_dataset()

    tuner = RandomSearch(
        build_model,
        objective=kt.Objective('val_accuracy', direction='max'),
        max_trials=1,
        directory=LOG_DIR
    )

    tuner.search(
        x=X_train,
        y=y_train,
        epochs=3,
        batch_size=64,
        validation_data=(X_test, y_test)
    )

    print(tuner.get_best_hyperparameters()[0].values)
    print('-----')
    print(tuner.get_best_models()[0].summary())


def train_cnn(time_split, time_interval):
    score = evaluate_model(time_split, time_interval)
    score = score * 100.0
    print('score:', score, '%')
