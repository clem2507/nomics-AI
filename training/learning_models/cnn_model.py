import os
import sys
import statistics
import numpy as np
import keras_tuner as kt

from math import sqrt
from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from keras_tuner.tuners import RandomSearch
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import save_model
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/training/data_loader')
sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/utils')

from preprocessing import Preprocessing
from util import plot_confusion_matrix, f1_m, num_of_correct_pred, TimingCallback


def evaluate_model(time_split, time_resampling, epochs, num_class):
    X_train, y_train, X_test, y_test = Preprocessing(time_split=time_split, time_resampling=time_resampling, num_class=num_class).create_dataset()
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    X_train, y_train = shuffle(X_train, y_train)
    validation_split, verbose, batch_size = 0.1, 1, 32
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    # model.add(Conv1D(filters=96, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # model.add(Dense(416, activation='relu'))
    if num_class == 2:
        # sigmoid activation function better than softmax for binary classification
        model.add(Dense(n_outputs, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
    else:
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
    print(model.summary())
    time_callback = TimingCallback()
    # fit network
    history = model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback])

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

    int_y_test = []
    for item in y_test:
        int_y_test.append(np.argmax(item))

    # for 95 % confidence interval
    interval = 1.96 * sqrt((accuracy * (1 - accuracy)) / len(X_test))
    lower_bound, upper_bound = proportion_confint(num_of_correct_pred(y_true=int_y_test, y_pred=classes), len(classes), 0.05)

    cm = confusion_matrix(y_true=int_y_test, y_pred=classes)

    if num_class == 2:
        cm_plt = plot_confusion_matrix(cm=cm, classes=['Invalid', 'Valid'], title='Confusion Matrix')
        model_type = 'binomial'
    else:
        cm_plt = plot_confusion_matrix(cm=cm, classes=['Invalid', 'Valid', 'Awake'], title='Confusion Matrix')
        model_type = 'multinomial'

    # Save the model
    if not os.path.exists(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/split_{time_split}_resampling_{time_resampling}'):
        os.mkdir(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/split_{time_split}_resampling_{time_resampling}')
    if not os.path.exists(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs'):
        os.mkdir(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs')
    if not os.path.exists(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs/ht'):
        os.mkdir(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs/ht')
    cm_plt.savefig(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs/cm_plt.png', bbox_inches="tight")
    cm_plt.close()
    # TODO Change the file path to make it depend on the current time it has been created??
    filepath = os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs/saved_model'
    model_info_file = open(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs/info.txt', 'w')
    model_info_file.write(f'This file contains information about the {num_class} classes CNN model accuracy using a {time_split} minutes signal time split and {time_resampling} minutes median data resampling \n')
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
    # In fact, if we repeated this experiment over and over, each time drawing a new sample S, containing […] new examples, we would find that for approximately 95% of these experiments, the calculated interval would contain the true error. For this reason, we call this interval the 95% confidence interval estimate
    model_info_file.write(f'Radius of the CI = {interval} \n')
    model_info_file.write(f'True classification of the model is likely between {accuracy - interval} and {accuracy + interval} \n')
    model_info_file.write(f'Lower bound model classification accuracy = {lower_bound} \n')
    model_info_file.write(f'Upper bound model classification accuracy = {upper_bound} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Test accuracy = {accuracy} \n')
    model_info_file.write(f'Confusion matrix (invalid | valid) = \n {cm} \n')
    model_info_file.write('--- \n')
    model_info_file.close()

    x = list(range(1, epochs + 1))

    figure, axes = plt.subplots(nrows=3, ncols=1)

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

    figure.set_figheight(17)
    figure.set_figwidth(12)
    figure.tight_layout()
    figure.savefig(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs/metrics_plt.png', bbox_inches='tight')
    plt.close(fig=figure)

    save_model(model, filepath)

    figure.tight_layout()

    return accuracy


def hyperparameters_tuning(time_split, time_resampling, max_trials, epochs, batch_size, num_class):
    def build_model(hp):
        n_timesteps, n_features, n_outputs = int(time_split * (60 / time_resampling)), 1, num_class
        model = Sequential()
        model.add(Conv1D(filters=hp.Int('input_number_filters', min_value=32, max_value=160, step=32), kernel_size=hp.Int('input_kernel size', min_value=2, max_value=5, step=1), activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Conv1D(filters=hp.Int('conv_layer_1_number_filters', min_value=32, max_value=160, step=32), kernel_size=hp.Int('conv_layer_1_kernel size', min_value=2, max_value=5, step=1), activation='relu'))
        model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(hp.Int('dense_layer_units', min_value=32, max_value=544, step=64), activation='relu'))
        if num_class == 2:
            # sigmoid activation function better than softmax for binary classification
            model.add(Dense(n_outputs, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
        else:
            model.add(Dense(n_outputs, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
        return model

    if num_class == 2:
        model_type = 'binomial'
    else:
        model_type = 'multinomial'

    LOG_DIR = os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/ht_tuning/results_split_{time_split}_resampling_{time_resampling}'
    X_train, y_train, X_test, y_test = Preprocessing(time_split=time_split, time_resampling=time_resampling, num_class=num_class).create_dataset()

    tuner = RandomSearch(
        build_model,
        objective=kt.Objective('val_accuracy', direction='max'),
        max_trials=max_trials,
        directory=LOG_DIR,
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

    ht_info_file = open(os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/cnn/{model_type}/ht_tuning/results_split_{time_split}_resampling_{time_resampling}/info.txt', 'w')
    ht_info_file.write(f'This file contains information about the CNN hyperparameters tuning \n')
    ht_info_file.write('--- \n')
    ht_info_file.write(f'Splitting time in minutes = {time_split} \n')
    ht_info_file.write(f'Resampling time in seconds = {time_resampling} \n')
    ht_info_file.write('--- \n')
    ht_info_file.write(f'Num of epochs = {epochs} \n')
    ht_info_file.write(f'Batch size = {batch_size} \n')
    ht_info_file.write(f'Num of random trials = {max_trials} \n')
    ht_info_file.write('--- \n')
    ht_info_file.write(f'Best hyperparameters found = {tuner.get_best_hyperparameters()[0].values} \n')
    ht_info_file.write('--- \n')
    ht_info_file.close()

    print('successfully saved!')


def train_cnn(time_split, time_resampling, epochs, num_class):
    time_split = float(time_split)
    time_resampling = float(time_resampling)
    score = evaluate_model(time_split, time_resampling, epochs, num_class)
    score = score * 100.0
    print('score:', score, '%')
    print('-----')
