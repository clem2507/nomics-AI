import os
import sys
import time
import datetime
import statistics
import numpy as np
import keras_tuner as kt

from math import sqrt
from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from keras_tuner.tuners import RandomSearch
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import save_model
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/training/data_loader')
sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/utils')

from preprocessing import Preprocessing
from util import plot_confusion_matrix, f1_m, num_of_correct_pred, TimingCallback


def evaluate_model(segmentation_value, downsampling_value, epochs, num_class, data_balancing, log_time):
    """
    Method used to create and evaluate a LSTM model on data

    Parameters:

    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -epochs: number of epochs to train the model
    -num_class: number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -data_balancing: true if balanced data is needed, false otherwise

    Returns:

    -accuracy: model accuracy on the testing set
    """

    X_train, y_train, X_test, y_test = Preprocessing(segmentation_value=segmentation_value, downsampling_value=downsampling_value, num_class=num_class, data_balancing=data_balancing, log_time=log_time).create_dataset()
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    X_train, y_train = shuffle(X_train, y_train)
    model = Sequential()
    validation_split, verbose, batch_size = 0.1, 1, 64
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    if num_class == 2:
        # sigmoid activation function better than softmax for binary classification
        model.add(Dense(n_outputs, activation='sigmoid'))
        # binary crossentropy loss function better than categorical crossentropy for binary classification
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
    else:
        # softmax activation function better than sigmoid for multinomial classification
        model.add(Dense(n_outputs, activation='softmax'))
        # categorical crossentropy loss function better than binary crossentropy for binary classification
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
    print(model.summary())
    time_callback = TimingCallback()
    # fit network
    history = model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback])

    # metrics history epochs after epochs
    training_accuracy_history = history.history['accuracy']
    training_f1_history = history.history['f1_m']
    training_loss_history = history.history['loss']
    validation_accuracy_history = history.history['val_accuracy']
    validation_f1_history = history.history['val_f1_m']
    validation_loss_history = history.history['val_loss']

    # evaluate model
    _, accuracy, *r = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

    predictions = model.predict(X_test)
    classes = np.argmax(predictions, axis=1)

    int_y_test = []    # predicted label list
    # loop that runs through the list of model predictions to keep the highest predicted probability values
    for item in y_test:
        int_y_test.append(np.argmax(item))

    # 95 % confidence interval computation
    interval = 1.96 * sqrt((accuracy * (1 - accuracy)) / len(X_test))
    lower_bound, upper_bound = proportion_confint(num_of_correct_pred(y_true=int_y_test, y_pred=classes), len(classes), 0.05)

    # confustion matrix np array creation based on the prediction made by the model on the test data
    cm = confusion_matrix(y_true=int_y_test, y_pred=classes)

    if num_class == 2:
        # confusion matrix plot
        cm_plt = plot_confusion_matrix(cm=cm, classes=['Invalid', 'Valid'], title='Confusion Matrix')
        model_type = 'binary'
    else:
        # confusion matrix plot
        cm_plt = plot_confusion_matrix(cm=cm, classes=['Invalid', 'Valid', 'Awake'], title='Confusion Matrix')
        model_type = 'multinomial'

    if data_balancing:
        is_balanced = 'balanced'
    else:
        is_balanced = 'unbalanced'

    # directories creation
    save_dir = os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/lstm/{log_time}'
    os.mkdir(save_dir)
    cm_plt.savefig(f'{save_dir}/cm_plt.png', bbox_inches="tight")
    cm_plt.close()
    model_info_file = open(f'{save_dir}/info.txt', 'w')
    model_info_file = open(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/{is_balanced}/lstm/{model_type}/split_{segmentation_value}_resampling_{downsampling_value}/{epochs}_epochs/info.txt', 'w')
    model_info_file.write(f'This file contains information about the {num_class} classes LSTM model accuracy using a {segmentation_value} minutes signal time split and {downsampling_value} minutes median data resampling \n')
    model_info_file.write(f'Signal resolution = {1/downsampling_value} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Is balanced = {is_balanced} \n')
    model_info_file.write(f'Model type = {model_type} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Num of epochs = {epochs} \n')
    model_info_file.write(f'Epochs training computation time (in sec) = {time_callback.logs} \n')
    model_info_file.write(f'Epochs training computation time mean (in sec) = {statistics.mean(time_callback.logs)} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Training accuracy history = {training_accuracy_history} \n')
    model_info_file.write(f'Training f1_score history = {training_f1_history} \n')
    model_info_file.write(f'Training loss history = {training_loss_history} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Validation accuracy history = {validation_accuracy_history} \n')
    model_info_file.write(f'Validation f1_score history = {validation_f1_history} \n')
    model_info_file.write(f'Validation loss history = {validation_loss_history} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Radius of the CI = {interval} \n')
    model_info_file.write(f'True classification of the model is likely between {accuracy - interval} and {accuracy + interval} \n')
    model_info_file.write(f'Lower bound model classification accuracy = {lower_bound} \n')
    model_info_file.write(f'Upper bound model classification accuracy = {upper_bound} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Test accuracy = {accuracy} \n')
    model_info_file.write(f'Confusion matrix (invalid | valid) = \n {cm} \n')
    model_info_file.write('--- \n')
    model_info_file.close()

    # chart with the learning curves creation
    figure, axes = plt.subplots(nrows=3, ncols=1)

    x = list(range(1, epochs+1))

    axes[0].plot(x, training_accuracy_history, label='train acc')
    axes[0].plot(x, validation_accuracy_history, label='val acc')
    axes[0].set_title('Training and validation accuracy')
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('accuracy')
    axes[0].legend(loc='best')
    axes[0].grid()

    axes[1].plot(x, training_f1_history, label='train f1')
    axes[1].plot(x, validation_f1_history, label='val f1')
    axes[1].set_title('Training and validation f1 score')
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('f1 score')
    axes[1].legend(loc='best')
    axes[1].grid()

    axes[2].plot(x, training_loss_history, label='train loss')
    axes[2].plot(x, validation_loss_history, label='val loss')
    axes[2].set_title('Training and validation loss')
    axes[2].set_xlabel('epochs')
    axes[2].set_ylabel('loss')
    axes[2].legend(loc='best')
    axes[2].grid()

    figure.set_figheight(10)
    figure.set_figwidth(6)
    figure.tight_layout()
    # plot save
    figure.savefig(f'{save_dir}/metrics_plt.png', bbox_inches='tight')

    plt.close(fig=figure)

    # save of the model weights
    save_model(model, f'{save_dir}/saved_model')

    return accuracy


def hyperparameters_tuning(segmentation_value, downsampling_value, max_trials, epochs, batch_size, num_class, data_balancing):
    """
    Method used to launch the hyperparameters tuning of the LSTM model

    Parameters:

    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -max_trials: number total of trials before outputting the best model
    -epochs: number of epochs to train the model
    -batch_size: number of instances from the training dataset used in the estimate of the error gradient
    -num_class: number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -data_balancing: true if balanced data is needed, false otherwise

    Returns:

    -accuracy: model accuracy on the testing set
    """

    def build_model(hp):
        n_timesteps, n_features, n_outputs = int(segmentation_value * (60 / downsampling_value)), 1, num_class
        model = Sequential()
        model.add(LSTM(hp.Int('input_units', min_value=32, max_value=544, step=64), input_shape=(n_timesteps, n_features)))
        model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        for i in range(hp.Int('n_layers', min_value=1, max_value=3, step=1)):
            model.add(Dense(hp.Int(f'dense_{i}_units', min_value=32, max_value=544, step=64), activation='relu'))
        if num_class == 2:
            model.add(Dense(n_outputs, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
        else:
            model.add(Dense(n_outputs, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
        return model

    if num_class == 2:
        model_type = 'binary'
    else:
        model_type = 'multinomial'

    if data_balancing:
        is_balanced = 'balanced'
    else:
        is_balanced = 'unbalanced'

    LOG_DIR = str(datetime.datetime.fromtimestamp(time.time()))
    save_dir = os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/lstm/ht/{LOG_DIR}'
    X_train, y_train, X_test, y_test = Preprocessing(segmentation_value=segmentation_value, downsampling_value=downsampling_value, num_class=num_class).create_dataset()

    tuner = RandomSearch(
        build_model,
        objective=kt.Objective('val_accuracy', direction='max'),
        max_trials=max_trials,
        directory=save_dir,
        overwrite=True,
        project_name='checkpoints'
    )

    tuner.search(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )

    ht_info_file = open(save_dir, 'w')
    ht_info_file.write(f'This file contains information about the LSTM hyperparameters tuning \n')
    ht_info_file.write('--- \n')
    ht_info_file.write(f'Model type = {model_type} \n')
    ht_info_file.write(f'Is balanced = {is_balanced} \n')
    ht_info_file.write('--- \n')
    ht_info_file.write(f'Splitting time in minutes = {segmentation_value} \n')
    ht_info_file.write(f'Resampling time in seconds = {downsampling_value} \n')
    ht_info_file.write('--- \n')
    ht_info_file.write(f'Num of epochs = {epochs} \n')
    ht_info_file.write(f'Batch size = {batch_size} \n')
    ht_info_file.write(f'Num of random trials = {max_trials} \n')
    ht_info_file.write('--- \n')
    ht_info_file.write(f'Best hyperparameters found = {tuner.get_best_hyperparameters()[0].values} \n')
    ht_info_file.write('--- \n')
    ht_info_file.close()

    print('successfully saved!')


def train_lstm(segmentation_value, downsampling_value, epochs, num_class, data_balancing):
    """
    Callable method to start the training of the LSTM model

    Parameters:

    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -epochs: number of epochs to train the model
    -num_class: number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -data_balancing: true if balanced data is needed, false otherwise
    """

    segmentation_value = float(segmentation_value)
    downsampling_value = float(downsampling_value)
    score = evaluate_model(segmentation_value, downsampling_value, epochs, num_class, data_balancing, log_time)
    score = score * 100.0
    print('score:', score, '%')
    print('-----')
