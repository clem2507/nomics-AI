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
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import save_model
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/training/data_loader')
sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/utils')

from preprocessing import Preprocessing
from util import plot_confusion_matrix, f1_m, num_of_correct_pred, stateful_fit, TimingCallback, ResetStatesCallback


def evaluate_model(analysis_directory, model_name, segmentation_value, downsampling_value, epochs, num_class, data_balancing, log_time, stateful=False):
    """
    Method used to create and evaluate a deep learning model on data, either CNN or LSTM

    Parameters:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -model_name: name of the model to train, either CNN or LSTM
    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -epochs: number of epochs to train the model
    -num_class: number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -data_balancing: true if balanced data is needed, false otherwise
    -log_time: save the time when the computation starts for the directory name
    -stateful: true to use stateful parameter for LSTM training, false to use stateless

    Returns:

    -accuracy: model accuracy on the testing set
    """

    model = Sequential()
    if model_name == 'cnn':
        X_train, y_train, X_test, y_test = Preprocessing(analysis_directory=analysis_directory, segmentation_value=segmentation_value, downsampling_value=downsampling_value, num_class=num_class, data_balancing=data_balancing, log_time=log_time).create_dataset_cnn()
        n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
        X_train, y_train = shuffle(X_train, y_train)
        validation_split, verbose, batch_size = 0.1, 1, 32
        if num_class == 2:
            # using 1Hz resolution and 1 minute window -- 60 sample size
            model.add(Conv1D(filters=16, kernel_size=5, strides=2, activation='relu', input_shape=(n_timesteps, n_features)))   # conv layer -- 1 -- Output size 16 x 29
            model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu'))    # conv layer -- 2 -- Output size 32 x 27
            model.add(MaxPooling1D(pool_size=2))   # max pooling -- 3 -- Output size 32 x 27

            model.add(Dropout(0.2))   # dropout -- 4 -- Output size 32 x 14

            model.add(Flatten())   # flatten layer -- 5 -- Output size 1 x 448
            model.add(Dense(64, activation='relu'))   # fully connected layer -- 6 -- Output size 1 x 64

            model.add(Dropout(0.2))   # dropout -- 7 -- Output size 1 x 64

            model.add(Dense(n_outputs, activation='sigmoid'))   # fully connected layer -- 8 -- Output size 1 x 2
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
        else:
            # using 10Hz resolution and 1 minute window -- 600 sample size
            model.add(Conv1D(filters=64, kernel_size=5, strides=2, activation='relu', input_shape=(n_timesteps, n_features)))   # conv layer -- 1 -- Output size 64 x 298
            model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='relu'))   # conv layer -- 2 -- Output size 128 x 294
            model.add(MaxPooling1D(pool_size=2))   # max pooling -- 3 -- Output size 128 x 147

            model.add(Dropout(0.2))   # dropout -- 4 -- Output size 128 x 147

            model.add(Conv1D(filters=128, kernel_size=13, strides=1, activation='relu'))   # conv layer -- 5 -- Output size 128 x 135
            model.add(Conv1D(filters=256, kernel_size=7, strides=1, activation='relu'))   # conv layer -- 6 -- Output size 256 x 129
            model.add(MaxPooling1D(pool_size=2))   # max pooling -- 7 -- Output size 256 x 64

            model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu'))   # conv layer -- 8 -- Output size 32 x 62
            model.add(Conv1D(filters=64, kernel_size=6, strides=1, activation='relu'))    # conv layer -- 9 -- Output size 64 x 57
            model.add(MaxPooling1D(pool_size=2))   # max pooling -- 10 -- Output size 64 x 28

            model.add(Conv1D(filters=8, kernel_size=5, strides=1, activation='relu'))   # conv layer -- 11 -- Output size 8 x 24
            model.add(Conv1D(filters=8, kernel_size=2, strides=1, activation='relu'))   # conv layer -- 12 -- Output size 8 x 23
            model.add(MaxPooling1D(pool_size=2))   # max pooling -- 13 -- Output size 8 x 11

            model.add(Flatten())   # flatten layer -- 14 -- Output size 1 x 88
            model.add(Dense(64, activation='relu'))   # fully connected layer -- 15 -- Output size 1 x 64

            model.add(Dropout(0.2))   # dropout -- 16 -- Output size 1 x 64

            model.add(Dense(n_outputs, activation='softmax'))   # fully connected layer -- 17 -- Output size 1 x 3
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
    elif model_name == 'lstm':
        # model.add(LSTM(128, activation='tanh', stateful=stateful, batch_input_shape=(batch_size, n_timesteps, n_features)))
        # model.add(Dropout(0.2))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.2))
        X_train, y_train, X_test, y_test = Preprocessing(analysis_directory=analysis_directory, segmentation_value=segmentation_value, downsampling_value=downsampling_value, num_class=num_class, data_balancing=data_balancing, log_time=log_time).create_dataset_lstm()
        n_timesteps, n_features, n_outputs = int(segmentation_value * (60 / downsampling_value)), 1, 2
        X_train, y_train = shuffle(X_train, y_train)
        batch_size = int(n_timesteps/2)
        if num_class == 2:
            model.add(LSTM(10, batch_input_shape=(1, 1, 1), stateful=stateful))
            # sigmoid activation function better than softmax for binary classification
            model.add(Dense(1, activation='sigmoid'))
            # binary crossentropy loss function better than categorical crossentropy for binary classification
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
        else:
            model.add(LSTM(50, batch_input_shape=(batch_size, n_timesteps, n_features), stateful=stateful))
            # softmax activation function better than sigmoid for multinomial classification
            model.add(Dense(n_outputs, activation='softmax'))
            # categorical crossentropy loss function better than binary crossentropy for binary classification
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
    else:
        raise Exception('model type does not exist, please choose between LSTM and CNN')
    
    print(model.summary())

    time_callback = TimingCallback()

    # directory creation
    save_dir = os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/{model_name}/{log_time}'
    os.mkdir(save_dir)

    # fit network
    if stateful and model_name == 'lstm':
        # reset_callback = ResetStatesCallback(max_len=n_timesteps)
        # history = model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback, reset_callback])
        model = stateful_fit(model, X_train, y_train, epochs)
    else:
        history = model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback])

        # metrics history epochs after epochs
        training_accuracy_history = history.history['accuracy']
        training_f1_history = history.history['f1_m']
        training_loss_history = history.history['loss']
        validation_accuracy_history = history.history['val_accuracy']
        validation_f1_history = history.history['val_f1_m']
        validation_loss_history = history.history['val_loss']

        # evaluate model
        _, accuracy, *r = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)

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
            labels = ['Invalid', 'Valid']
            cm_plt = plot_confusion_matrix(cm=cm, classes=labels, title='Confusion Matrix')
            model_type = 'binary'
        else:
            # confusion matrix plot
            labels = ['Invalid', 'Valid', 'Awake']
            cm_plt = plot_confusion_matrix(cm=cm, classes=labels, title='Confusion Matrix')
            model_type = 'multinomial'

        if data_balancing:
            is_balanced = 'balanced'
        else:
            is_balanced = 'unbalanced'

        cm_plt.savefig(f'{save_dir}/cm_plt.png', bbox_inches='tight')
        cm_plt.close()
        model_info_file = open(f'{save_dir}/info.txt', 'w')
        model_info_file.write(f'This file contains information about the {model_name} model \n')
        model_info_file.write('--- \n')
        model_info_file.write(f'Num of classes = {num_class} \n')
        model_info_file.write(f'Segmentation value = {segmentation_value} \n')
        model_info_file.write(f'Downsampling value = {downsampling_value} \n')
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
        model_info_file.write(f'Training f1 score history = {training_f1_history} \n')
        model_info_file.write(f'Training loss history = {training_loss_history} \n')
        model_info_file.write('--- \n')
        model_info_file.write(f'Validation accuracy history = {validation_accuracy_history} \n')
        model_info_file.write(f'Validation f1 score history = {validation_f1_history} \n')
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

        x = list(range(1, epochs + 1))

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


def hyperparameters_tuning(model_name, segmentation_value, downsampling_value, max_trials, epochs, batch_size, num_class, data_balancing):
    """
    Method used to launch the hyperparameters tuning of the model

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
        if model_name == 'cnn':
            model.add(Conv1D(filters=hp.Int('input_number_filters', min_value=32, max_value=160, step=32), kernel_size=hp.Int('input_kernel size', min_value=2, max_value=5, step=1), activation='relu', input_shape=(n_timesteps, n_features)))
            model.add(Conv1D(filters=hp.Int('conv_layer_1_number_filters', min_value=32, max_value=160, step=32), kernel_size=hp.Int('conv_layer_1_kernel size', min_value=2, max_value=5, step=1), activation='relu'))
            model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(hp.Int('dense_layer_units', min_value=32, max_value=544, step=64), activation='relu'))
            if num_class == 2:
                model.add(Dense(n_outputs, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
            else:
                model.add(Dense(n_outputs, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
            return model
        elif model_name == 'lstm':
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
        else:
            raise Exception('model type does not exist, please choose between LSTM and CNN')

    if num_class == 2:
        model_type = 'binary'
    else:
        model_type = 'multinomial'

    if data_balancing:
        is_balanced = 'balanced'
    else:
        is_balanced = 'unbalanced'

    LOG_DIR = str(datetime.datetime.fromtimestamp(time.time()))
    save_dir = os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/{model_name}/ht/{LOG_DIR}'
    X_train, y_train, X_test, y_test = Preprocessing(segmentation_value=segmentation_value, downsampling_value=downsampling_value, num_class=num_class, data_balancing=data_balancing, log_time=LOG_DIR).create_dataset()

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
    ht_info_file.write(f'This file contains information about the {model_name} hyperparameters tuning \n')
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


def train_model(analysis_directory, model, segmentation_value, downsampling_value, epochs, num_class, data_balancing, log_time, stateful):
    """
    Callable method to start the training of the model

    Parameters:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -model: name of the model to train, either CNN or LSTM
    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -epochs: number of epochs to train the model
    -num_class: number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -data_balancing: true if balanced data is needed, false otherwise
    -log_time: save the time when the computation starts for the directory name
    -stateful: true to use stateful parameter for LSTM training, false to use stateless
    """

    segmentation_value = float(segmentation_value)
    downsampling_value = float(downsampling_value)
    score = evaluate_model(analysis_directory, model, segmentation_value, downsampling_value, epochs, num_class, data_balancing, log_time, stateful)
    score = score * 100.0
    print('score:', score, '%')
    print('-----')
