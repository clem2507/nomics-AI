import os
import sys
import time
import itertools
import statistics
import numpy as np

from math import sqrt
from tensorflow.keras.models import save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/training/data_loader')
sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')

from preprocessing import Preprocessing
from util import reduction, plot_confusion_matrix, TimingCallback, f1_m, multilabel_to_onelabel


def evaluate_model(analysis_directory, model_name, segmentation_value, downsampling_value, epochs, data_balancing, log_time, batch_size, standard_scale, sliding_window, stateful, center_of_interest, task, full_sequence):
    """
    Method used to create and evaluate a deep learning model on data, either CNN or LSTM

    Parameters:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -model_name: name of the model to train, either CNN or LSTM
    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -epochs: number of epochs to train the model
    -data_balancing: true if balanced data is needed, false otherwise
    -log_time: save the time when the computation starts for the directory name
    -batch_size: batch size for training
    -standard_scale: true to perform a standardizatin by centering and scaling the data
    -stateful: true to use stateful LSTM instead of stateless
    -sliding_window: true to use sliding window with a small center portion of interest
    -center_of_interest: center of interest size in seconds for the sliding window
    -task: corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep
    -full_sequence: true to feed the entire sequence without dividing it into multiple windows

    Returns:

    -accuracy: model accuracy on the testing set
    """

    # directory creation
    save_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task{task}/{model_name}/{log_time}'
    os.mkdir(save_dir)
    if not stateful:
        os.mkdir(f'{save_dir}/best')

        checkpoint_filepath = f'{save_dir}/best'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )

    X_train, y_train, X_test, y_test = Preprocessing(analysis_directory=analysis_directory, segmentation_value=segmentation_value, downsampling_value=downsampling_value, data_balancing=data_balancing, log_time=log_time, standard_scale=standard_scale, sliding_window=sliding_window, stateful=stateful, center_of_interest=center_of_interest, task=task, full_sequence=full_sequence).create_dataset()

    model = Sequential()
    # Stateless model
    if not stateful:
        if not full_sequence:
            n_timesteps, n_features, n_outputs, validation_split, verbose = X_train.shape[1], X_train.shape[2], y_train.shape[1], 0.1, 1
        else:
            n_timesteps, n_features, n_outputs, validation_split, verbose = None, len(X_train[0]), len(y_train[0]), 0.1, 1
        if model_name == 'cnn':
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
        elif model_name == 'lstm':
            # model.add(LSTM(500, input_shape=(n_timesteps, n_features)))   # lstm layer -- 1
            # model.add(Dropout(0.2))   # dropout -- 2
            # model.add(Dense(500, activation='relu'))   # fully connected layer -- 3
            # model.add(Dropout(0.2))   # dropout -- 4
            # model.add(Dense(200, activation='relu'))   # fully connected layer -- 5
            # model.add(Dropout(0.2))   # dropout -- 6
            # model.add(Dense(n_outputs, activation='sigmoid'))   # fully connected layer -- 7
            
            model.add(LSTM(20, input_shape=(n_timesteps, n_features)))   # lstm layer -- 1
            model.add(Dense(20, activation='relu'))   # fully connected layer -- 2
            model.add(Dropout(0.2))   # dropout -- 3
            model.add(Dense(10, activation='relu'))   # fully connected layer -- 4
            model.add(Dropout(0.2))   # dropout -- 5
            model.add(Dense(n_outputs, activation='sigmoid'))   # fully connected layer -- 6

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])

        print(model.summary())

        time_callback = TimingCallback()

        history = model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback, model_checkpoint_callback])

        computation_time_history = time_callback.logs

        # metrics history epochs after epochs
        training_loss_history = history.history['loss']
        training_accuracy_history = history.history['accuracy']
        training_f1_history = history.history['f1_m']
        validation_loss_history = history.history['val_loss']
        validation_accuracy_history = history.history['val_accuracy']
        validation_f1_history = history.history['f1_m']

        # evaluate model
        dic = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose, return_dict=True)

        te_accuracy = dic['accuracy']
        te_f1 = dic['f1_m']

        predictions = model.predict(X_test)
        classes = np.argmax(predictions, axis=1)

        total_y_test = []
        for item in y_test:
            total_y_test.append(np.argmax(item))
        y_test = total_y_test
    # Stateful model
    else:
        max_length = int(segmentation_value * (60 / downsampling_value))
        n_timesteps, n_features, n_outputs, validation_split, return_sequences = max_length, 1, 2, 0.1, True
        # model.add(LSTM(500, batch_input_shape=(batch_size, max_length, n_features), return_sequences=return_sequences, stateful=True))   # lstm layer -- 1
        # model.add(Dropout(0.2))   # dropout -- 2
        # model.add(Dense(500, activation='relu'))   # fully connected layer -- 3
        # model.add(Dropout(0.2))   # dropout -- 4
        # model.add(Dense(200, activation='relu'))   # fully connected layer -- 5
        # model.add(Dropout(0.2))   # dropout -- 6
        # model.add(Dense(n_outputs, activation='sigmoid'))   # fully connected layer -- 7

        if full_sequence:
            n_timesteps = None

        model.add(LSTM(20, batch_input_shape=(batch_size, n_timesteps, n_features), return_sequences=return_sequences, stateful=True))   # lstm layer -- 1
        model.add(Dense(20, activation='relu'))   # fully connected layer -- 2
        model.add(Dropout(0.2))   # dropout -- 3
        model.add(Dense(10, activation='relu'))   # fully connected layer -- 4
        model.add(Dropout(0.2))   # dropout -- 5
        model.add(Dense(n_outputs, activation='sigmoid'))   # fully connected layer -- 6

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1_m])

        print(model.summary())

        training_loss_history = []
        training_accuracy_history = []
        training_f1_history = []
        validation_loss_history = []
        validation_accuracy_history = []
        validation_f1_history = []

        computation_time_history = []
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, random_state=42)

        print(f'# train samples = {len(X_train)}')
        print(f'# val samples = {len(X_val)}')
        print(f'# test samples = {len(X_test)}')

        # Stateful implementation with train_on_batch
        print('Model train...')
        for epoch in range(epochs):
            print('----- EPOCH #{} -----'.format(epoch+1))
            print('---> Training')
            start = time.time()
            mean_tr_loss = []
            mean_tr_acc = []
            mean_tr_f1 = []
            for i in tqdm(range(len(X_train))):
                if not full_sequence:
                    for j in range(0, len(X_train[i]), batch_size*n_timesteps):
                        if j+(batch_size*n_timesteps) < len(X_train[i]):
                            dic = model.train_on_batch(np.reshape(X_train[i][j:j+(batch_size*n_timesteps)], (batch_size, n_timesteps, n_features)), np.reshape([y_train[i][j:j+(batch_size*n_timesteps)]], (batch_size, n_timesteps, n_outputs)), return_dict=True)
                            mean_tr_loss.append(dic['loss'])
                            mean_tr_acc.append(dic['acc'])
                            mean_tr_f1.append(dic['f1_m'])
                else:
                    dic = model.train_on_batch(np.reshape(X_train[i], (batch_size, -1, n_features)), np.reshape(y_train[i], (batch_size, -1, n_outputs)), return_dict=True)
                    mean_tr_loss.append(dic['loss'])
                    mean_tr_acc.append(dic['acc'])
                    mean_tr_f1.append(dic['f1_m'])
                model.reset_states()

            training_loss_history.append(np.mean(mean_tr_loss))
            training_accuracy_history.append(np.mean(mean_tr_acc))
            training_f1_history.append(np.mean(mean_tr_f1))
            print('train loss = {}'.format(np.mean(mean_tr_loss)))
            print('train accuracy = {}'.format(np.mean(mean_tr_acc)))
            print('train f1 = {}'.format(np.mean(mean_tr_f1)))

            print('---> Validation')
            mean_val_loss = []
            mean_val_acc = []
            mean_val_f1 = []
            for i in tqdm(range(len(X_val))):
                if not full_sequence:
                    for j in range(0, len(X_val[i]), batch_size*n_timesteps):
                        if j+(batch_size*n_timesteps) < len(X_val[i]):
                            dic = model.test_on_batch(np.reshape(X_val[i][j:j+(batch_size*n_timesteps)], (batch_size, n_timesteps, n_features)), np.reshape([y_val[i][j:j+(batch_size*n_timesteps)]], (batch_size, n_timesteps, n_outputs)), return_dict=True)
                            mean_val_loss.append(dic['loss'])
                            mean_val_acc.append(dic['acc'])
                            mean_val_f1.append(dic['f1_m'])
                else:
                    dic = model.train_on_batch(np.reshape(X_val[i], (batch_size, -1, n_features)), np.reshape(y_val[i], (batch_size, -1, n_outputs)), return_dict=True)
                    mean_val_loss.append(dic['loss'])
                    mean_val_acc.append(dic['acc'])
                    mean_val_f1.append(dic['f1_m'])
                model.reset_states()

            validation_loss_history.append(np.mean(mean_val_loss))
            validation_accuracy_history.append(np.mean(mean_val_acc))
            validation_f1_history.append(np.mean(mean_val_f1))
            print('val loss = {}'.format(np.mean(mean_val_loss)))
            print('val accuracy = {}'.format(np.mean(mean_val_acc)))
            print('val f1 = {}'.format(np.mean(mean_val_f1)))

            computation_time_history.append(time.time()-start)

        print('---> Testing')
        total_classes = []
        total_y_test = []
        mean_te_acc = []
        mean_te_f1 = []
        for i in tqdm(range(len(X_test))):
            classes = []
            if not full_sequence:
                for j in range(0, len(X_test[i]), batch_size*n_timesteps):
                    if j+(batch_size*n_timesteps) < len(X_test[i]):
                        y_pred, *r = model.predict_on_batch(np.reshape(X_test[i][j:j+(batch_size*n_timesteps)], (batch_size, n_timesteps, n_features)))
                        for label in y_pred:
                            idx = np.argmax(label)
                            pred_label = idx
                            classes.append(pred_label)
                            total_classes.append(pred_label)
                total_y_test.append(y_test[i][:len(classes)])
                mean_te_acc.append(accuracy_score(y_test[i][:len(classes)], classes))
                mean_te_f1.append(f1_score(y_test[i][:len(classes)], classes))
            else:
                y_pred, *r = model.predict_on_batch(np.reshape(X_test[i], (batch_size, -1, n_features)))
                for label in y_pred:
                    idx = np.argmax(label)
                    pred_label = idx
                    classes.append(pred_label)
                    total_classes.append(pred_label)
                total_y_test.append(multilabel_to_onelabel(y_test[i]))
                mean_te_acc.append(accuracy_score(multilabel_to_onelabel(y_test[i]), classes))
                mean_te_f1.append(f1_score(multilabel_to_onelabel(y_test[i]), classes))
            model.reset_states()

        te_accuracy = np.mean(mean_te_acc)
        te_f1 = np.mean(mean_te_f1)
        classes = total_classes
        y_test = list(itertools.chain.from_iterable(total_y_test))


    # 95 % confidence interval computation
    interval = 1.96 * sqrt((te_accuracy * (1 - te_accuracy)) / len(X_test))

    # confustion matrix np array creation based on the prediction made by the model on the test data
    cm = confusion_matrix(y_true=y_test, y_pred=classes)

    # summary txt file
    model_info_file = open(f'{save_dir}/info.txt', 'w')
    model_info_file.write(f'This file contains information about the {model_name} model \n')
    model_info_file.write(f'Analysis directory = {analysis_directory} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Segmentation value = {segmentation_value} \n')
    model_info_file.write(f'Downsampling value = {downsampling_value} \n')
    model_info_file.write(f'Signal frequency = {1/downsampling_value} Hz \n')
    model_info_file.write(f'Batch size = {batch_size} \n')
    model_info_file.write(f'Standard scale = {int(standard_scale)} \n')
    model_info_file.write(f'Stateful = {int(stateful)} \n')
    model_info_file.write(f'Sliding window = {int(sliding_window)} \n')
    model_info_file.write(f'Center of interest = {center_of_interest} \n')
    model_info_file.write(f'Full sequence = {int(full_sequence)} \n')
    model_info_file.write(f'Data balancing = {int(data_balancing)} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Log time = {log_time} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Num of epochs = {epochs} \n')
    model_info_file.write(f'Epochs training computation time history (in sec) = {computation_time_history} \n')
    model_info_file.write(f'Epochs training computation time mean (in sec) = {statistics.mean(computation_time_history)} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Training loss history = {training_loss_history} \n')
    model_info_file.write(f'Training accuracy history = {training_accuracy_history} \n')
    model_info_file.write(f'Training f1 history = {training_f1_history} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Validation loss history = {validation_loss_history} \n')
    model_info_file.write(f'Validation accuracy history = {validation_accuracy_history} \n')
    model_info_file.write(f'Validation f1 history = {validation_f1_history} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Radius of the CI = {interval} \n')
    model_info_file.write(f'True classification of the model is likely between {te_accuracy - interval} and {te_accuracy + interval} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Test accuracy = {te_accuracy} \n')
    model_info_file.write(f'Confusion matrix (invalid | valid) = \n {cm} \n')
    model_info_file.write('--- \n')
    model_info_file.close()

    # confusion matrix plot
    if task == 1:
        labels = ['Invalid', 'Valid']
    elif task == 2:
        labels = ['Awake', 'Sleep']
    
    cm_plt = plot_confusion_matrix(cm=cm, classes=labels, title='Confusion Matrix', normalize=True)

    cm_plt.savefig(f'{save_dir}/cm_plt.png', bbox_inches='tight')
    cm_plt.close()

    # chart with the learning curves creation
    figure, axes = plt.subplots(nrows=3, ncols=1)

    x = list(range(1, epochs + 1))

    axes[0].plot(x, training_loss_history, label='train loss')
    axes[0].plot(x, validation_loss_history, label='val loss')
    axes[0].set_title('Training and validation loss')
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('loss')
    axes[0].legend(loc='best')
    axes[0].grid()

    axes[1].plot(x, training_accuracy_history, label='train acc')
    axes[1].plot(x, validation_accuracy_history, label='val acc')
    axes[1].set_title('Training and validation accuracy')
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('accuracy')
    axes[1].legend(loc='best')
    axes[1].grid()

    axes[2].plot(x, training_f1_history, label='train f1')
    axes[2].plot(x, validation_f1_history, label='val f1')
    axes[2].set_title('Training and validation f1')
    axes[2].set_xlabel('epochs')
    axes[2].set_ylabel('f1')
    axes[2].legend(loc='best')
    axes[2].grid()

    figure.set_figheight(12)
    figure.set_figwidth(6)
    figure.tight_layout()
    # plot save
    figure.savefig(f'{save_dir}/metrics_plt.png', bbox_inches='tight')

    plt.close(fig=figure)

    # save of the model weights
    os.mkdir(f'{save_dir}/last')
    save_model(model, f'{save_dir}/last')

    return te_accuracy, te_f1


def train_model(analysis_directory, model, segmentation_value, downsampling_value, epochs, data_balancing, log_time, batch_size, standard_scale, sliding_window, stateful, center_of_interest, task, full_sequence):
    """
    Callable method to start the training of the model

    Parameters:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -model: name of the model to train, either CNN or LSTM
    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -epochs: number of epochs to train the model
    -data_balancing: true if balanced data is needed, false otherwise
    -log_time: save the time when the computation starts for the directory name
    -batch_size: batch size for training
    -standard_scale: true to perform a standardizatin by centering and scaling the data
    -stateful: true to use stateful LSTM instead of stateless
    -sliding_window: true to use sliding window with a small center portion of interest
    -center_of_interest: center of interest size in seconds for the sliding window
    -task: corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep
    -full_sequence: true to feed the entire sequence without dividing it into multiple windows
    """

    segmentation_value = float(segmentation_value)
    downsampling_value = float(downsampling_value)
    te_accuracy, te_f1 = evaluate_model(analysis_directory, model, segmentation_value, downsampling_value, epochs, data_balancing, log_time, batch_size, standard_scale, sliding_window, stateful, center_of_interest, task, full_sequence)
    print('test accuracy:', te_accuracy*100, '%')
    print('test f1:', te_f1*100, '%')
    print('-----')
