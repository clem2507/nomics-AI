import wandb
from wandb.keras import WandbCallback

import os
import sys
import time
import itertools
import statistics
import numpy as np

from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, MaxPooling1D, Flatten, Dense, Activation, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/training/data_loader')
sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')

from preprocessing import Preprocessing
from util import reduction, plot_confusion_matrix, TimingCallback, f1


def evaluate_model(analysis_directory, model_name, segmentation_value, downsampling_value, epochs, data_balancing, log_time, batch_size, patience, standard_scale, sliding_window, stateful, center_of_interest, task, full_sequence, return_sequences):
    """
    Method used to create and evaluate a deep learning model on data, either CNN or LSTM

    Parameters:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -model_name: name of the model to train, either CNN or LSTM
    -segmentation_value: window segmentation value in second
    -downsampling_value: signal downsampling value in second
    -epochs: number of epochs to train the model
    -data_balancing: true if balanced data is needed, false otherwise
    -log_time: save the time when the computation starts for the directory name
    -batch_size: batch size for training
    -patience: number of epochs without learning before early stop the training
    -standard_scale: true to perform a standardizatin by centering and scaling the data
    -stateful: true to use stateful LSTM instead of stateless
    -sliding_window: true to use sliding window with a small center portion of interest
    -center_of_interest: center of interest size in seconds for the sliding window
    -task: corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep
    -full_sequence: true to feed the entire sequence without dividing it into multiple windows
    -return_sequences (--return_sequences): true to return the state of each data point in the full sequence for the LSTM model

    Returns:

    -accuracy: model accuracy on the testing set
    """

    wandb.init(project="nomics-AI", entity="nomics")

    wandb.config = {
        "epochs": epochs,
        "batch_size": batch_size
    }

    # directory creation
    save_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task{task}/{model_name}/{log_time}'
    os.mkdir(save_dir)
    os.mkdir(f'{save_dir}/best')
    checkpoint_filepath = f'{save_dir}/best/model-best.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='max',
        save_best_only=True
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
        restore_best_weights=True,
        min_delta=1e-3
    )

    X_train, y_train, X_test, y_test = Preprocessing(analysis_directory=analysis_directory, segmentation_value=segmentation_value, downsampling_value=downsampling_value, data_balancing=data_balancing, log_time=log_time, standard_scale=standard_scale, sliding_window=sliding_window, stateful=stateful, center_of_interest=center_of_interest, task=task, full_sequence=full_sequence, return_sequences=return_sequences).create_dataset()

    model = Sequential()
    # Stateless model
    if not stateful:
        if not full_sequence:
            n_timesteps, n_features, n_outputs, validation_split, verbose = X_train.shape[1], 1, 2, 0.1, 1
        else:
            n_timesteps, n_features, n_outputs, validation_split, verbose = None, 1, 2, 0.1, 1
        
        if model_name == 'cnn':
            # using 1Hz resolution and 1 minute window -- 60 sample size
            model.add(Conv1D(filters=16, kernel_size=5, strides=2, activation='relu', input_shape=(n_timesteps, n_features)))   # conv layer -- 1 -- Output size 16 x 29
            model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu'))    # conv layer -- 2 -- Output size 32 x 27
            model.add(MaxPooling1D(pool_size=2))   # max pooling -- 3 -- Output size 32 x 27
            model.add(Dropout(0.2))   # dropout -- 4 -- Output size 32 x 14
            model.add(Flatten())   # flatten layer -- 5 -- Output size 1 x 448
            model.add(Dense(64, activation='relu'))   # fully connected layer -- 6 -- Output size 1 x 64
            model.add(Dropout(0.2))   # dropout -- 7 -- Output size 1 x 64
            model.add(Dense(32, activation='relu'))   # fully connected layer -- 8 -- Output size 1 x 32
            model.add(Dropout(0.2))   # dropout -- 9 -- Output size 1 x 32
            model.add(Dense(n_outputs))   # fully connected layer -- 10 -- Output size 1 x 2
            model.add(Activation(activation='softmax'))   # activation layer -- 11 -- Output size 1 x 2
        elif model_name == 'lstm':
            stf = False
            lstm_units = max(24, int(2/3 * (int(segmentation_value * (1 / downsampling_value)) * n_outputs)))   # https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
            if task == 1:
                model.add(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=stf, input_shape=(n_timesteps, n_features)))   # lstm layer -- 1
            else:
                model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=stf), input_shape=(n_timesteps, n_features)))   # lstm layer -- 1
            if return_sequences:
                if task == 1:
                    model.add(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=stf))   # lstm layer -- 1'
                else:
                    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=stf)))   # lstm layer -- 1'
            model.add(Dropout(0.2))   # dropout -- 2
            model.add(Dense(lstm_units//2, activation='relu'))   # fully connected layer -- 3
            model.add(Dropout(0.2))   # dropout -- 4
            model.add(Dense(lstm_units//4, activation='relu'))   # fully connected layer -- 5
            model.add(Dropout(0.2))   # dropout -- 6
            model.add(Dense(units=n_outputs))   # fully connected layer -- 7
            model.add(Activation('softmax'))   # activation -- 8

        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])

        print(model.summary())

        time_callback = TimingCallback()

        history = model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback, model_checkpoint_callback, early_stop, WandbCallback()])

        computation_time_history = time_callback.logs

        # metrics history epochs after epochs
        training_loss_history = history.history['loss']
        training_accuracy_history = history.history['accuracy']
        training_f1_history = history.history['f1']
        validation_loss_history = history.history['val_loss']
        validation_accuracy_history = history.history['val_accuracy']
        validation_f1_history = history.history['val_f1']

        # evaluate model
        dic = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose, return_dict=True)

        te_accuracy = dic['accuracy']
        te_f1 = dic['f1']

        predictions = model.predict(X_test)
        if not return_sequences:
            classes = np.argmax(predictions, axis=1)
        else:
            classes = np.argmax(predictions, axis=2)

        total_y_test = []
        for item in y_test:
            if not return_sequences:
                total_y_test.append(np.argmax(item))
            else:
                total_y_test.append(np.argmax(item, axis=1))
        if not return_sequences:
            y_test = total_y_test
        else:
            y_test = list(itertools.chain.from_iterable(total_y_test))

    # Stateful model
    else:
        max_length = int(segmentation_value * (1 / downsampling_value))
        n_features, n_outputs, validation_split, return_sequences = 1, 1, 0.1, False
        if full_sequence:
            n_timesteps = None
            return_sequences = True
            lstm_units = 500
        else:
            n_timesteps = max_length
            lstm_units = max(8, int(2/3 * (int(segmentation_value * (1 / downsampling_value)) * n_outputs)))   # https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
            
        model.add(LSTM(units=lstm_units, batch_input_shape=(batch_size, n_timesteps, n_features), return_sequences=return_sequences, stateful=True))   # lstm layer -- 1
        model.add(Dropout(0.2))   # dropout -- 2
        model.add(Dense(lstm_units//2, activation='relu'))   # fully connected layer -- 3
        model.add(Dropout(0.2))   # dropout -- 4
        model.add(Dense(lstm_units//4, activation='relu'))   # fully connected layer -- 5
        model.add(Dropout(0.2))   # dropout -- 6
        model.add(Dense(units=n_outputs))   # fully connected layer -- 7
        model.add(Activation('sigmoid'))   # activation -- 8

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

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

        best_val_loss = 1
        patience_count = 0

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
                            if return_sequences:
                                dic = model.train_on_batch(np.reshape(X_train[i][j:j+(batch_size*n_timesteps)], (batch_size, n_timesteps, n_features)), np.reshape([y_train[i][j:j+(batch_size*n_timesteps)]], (batch_size, n_timesteps, n_outputs)), return_dict=True)
                            else:
                                dic = model.train_on_batch(np.reshape(X_train[i][j:j+(batch_size*n_timesteps)], (batch_size, n_timesteps, n_features)), np.reshape([max(y_train[i][j:j+step], key=y_train[i][j:j+step].count) for step in range(n_timesteps, (batch_size*n_timesteps)+1, n_timesteps)], (batch_size, 1, n_outputs)), return_dict=True)
                            mean_tr_loss.append(dic['loss'])
                            mean_tr_acc.append(dic['acc'])
                            mean_tr_f1.append(dic['f1'])
                else:
                    dic = model.train_on_batch(np.reshape(X_train[i], (batch_size, -1, n_features)), np.reshape(y_train[i], (batch_size, -1, n_outputs)), return_dict=True)
                    mean_tr_loss.append(dic['loss'])
                    mean_tr_acc.append(dic['acc'])
                    mean_tr_f1.append(dic['f1'])
                model.reset_states()

            tr_loss = np.mean(mean_tr_loss)
            tr_acc = np.mean(mean_tr_acc)
            tr_f1 = np.mean(mean_tr_f1)
            training_loss_history.append(tr_loss)
            training_accuracy_history.append(tr_acc)
            training_f1_history.append(tr_f1)
            print('train loss = {}'.format(tr_loss))
            print('train accuracy = {}'.format(tr_acc))
            print('train f1 = {}'.format(tr_f1))

            print('---> Validation')
            mean_val_loss = []
            mean_val_acc = []
            mean_val_f1 = []
            for i in tqdm(range(len(X_val))):
                if not full_sequence:
                    for j in range(0, len(X_val[i]), batch_size*n_timesteps):
                        if j+(batch_size*n_timesteps) < len(X_val[i]):
                            if return_sequences:
                                dic = model.test_on_batch(np.reshape(X_val[i][j:j+(batch_size*n_timesteps)], (batch_size, n_timesteps, n_features)), np.reshape([y_val[i][j:j+(batch_size*n_timesteps)]], (batch_size, n_timesteps, n_outputs)), return_dict=True)
                            else:
                                dic = model.test_on_batch(np.reshape(X_val[i][j:j+(batch_size*n_timesteps)], (batch_size, n_timesteps, n_features)), np.reshape([max(y_val[i][j:j+step], key=y_val[i][j:j+step].count) for step in range(n_timesteps, (batch_size*n_timesteps)+1, n_timesteps)], (batch_size, 1, n_outputs)), return_dict=True)
                            mean_val_loss.append(dic['loss'])
                            mean_val_acc.append(dic['acc'])
                            mean_val_f1.append(dic['f1'])
                else:
                    dic = model.train_on_batch(np.reshape(X_val[i], (batch_size, -1, n_features)), np.reshape(y_val[i], (batch_size, -1, n_outputs)), return_dict=True)
                    mean_val_loss.append(dic['loss'])
                    mean_val_acc.append(dic['acc'])
                    mean_val_f1.append(dic['f1'])
                model.reset_states()

            val_loss = np.mean(mean_val_loss)
            val_acc = np.mean(mean_val_acc)
            val_f1 = np.mean(mean_val_f1)
            validation_loss_history.append(val_loss)
            validation_accuracy_history.append(val_acc)
            validation_f1_history.append(val_f1)
            print('val loss = {}'.format(val_loss))
            print('val accuracy = {}'.format(val_acc))
            print('val f1 = {}'.format(val_f1))

            wandb.log({'loss': tr_loss, 'accuracy': tr_acc, 'f1': tr_f1, 'val_loss': val_loss, 'val_accuracy': val_acc, 'val_f1': val_f1})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(f'{save_dir}/best/model-best.h5')
                model.save(os.path.join(wandb.run.dir, "model-best.h5"))
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                break

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
                            if return_sequences:
                                pred_label = round(label[0])
                            else:
                                pred_label = round(label)
                            classes.append(pred_label)
                            total_classes.append(pred_label)
                reducted_y_test = reduction(y_test[i], batch_size, n_timesteps)
                total_y_test.append(reducted_y_test)
                mean_te_acc.append(accuracy_score(reducted_y_test, classes))
                mean_te_f1.append(f1_score(reducted_y_test, classes))
            else:
                y_pred, *r = model.predict_on_batch(np.reshape(X_test[i], (batch_size, -1, n_features)))
                for label in y_pred:
                    pred_label = round(label[0])
                    classes.append(pred_label)
                    total_classes.append(pred_label)
                total_y_test.append(y_test[i])
                mean_te_acc.append(accuracy_score(y_test[i], classes))
                mean_te_f1.append(f1_score(y_test[i], classes))
            model.reset_states()

        te_accuracy = np.mean(mean_te_acc)
        te_f1 = np.mean(mean_te_f1)
        classes = total_classes
        y_test = list(itertools.chain.from_iterable(total_y_test))


    # 95 % confidence interval computation
    interval = 1.96 * sqrt((te_accuracy * (1 - te_accuracy)) / len(X_test))

    # confusion matrix np array creation based on the prediction made by the model on the test data
    if not return_sequences:
        cm = confusion_matrix(y_true=y_test, y_pred=classes)
    else:
        cm = confusion_matrix(y_true=y_test, y_pred=list(itertools.chain.from_iterable(classes)))

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
    model_info_file.write(f'Return sequences = {int(return_sequences)} \n')
    model_info_file.write(f'Data balancing = {int(data_balancing)} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Log time = {log_time} \n')
    model_info_file.write('--- \n')
    model_info_file.write(f'Num of epochs = {epochs}, early stop after {len(training_loss_history)} epochs \n')
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
    model_info_file.write(f'Confusion matrix (0 | 1) = \n {cm} \n')
    model_info_file.write('--- \n')
    model_info_file.close()

    # confusion matrix plot
    if task == 1:
        labels = ['Invalid', 'Valid']
    elif task == 2:
        labels = ['Awake', 'Sleep']
    
    cm_plt = plot_confusion_matrix(cm=cm, classes=labels, title='Confusion Matrix', normalize=True)

    cm_plt.savefig(f'{save_dir}/cm_plt.png', bbox_inches='tight')
    wandb.log({'confusion_matrix': cm_plt})
    cm_plt.close()

    # chart with the learning curves creation
    figure, axes = plt.subplots(nrows=3, ncols=1)

    x = list(range(1, len(training_loss_history) + 1))

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
    model.save(f'{save_dir}/last/model-last.h5')
    model.save(os.path.join(wandb.run.dir, "model-last.h5"))

    return te_accuracy, te_f1


def train_model(analysis_directory, model, segmentation_value, downsampling_value, epochs, data_balancing, log_time, batch_size, patience, standard_scale, sliding_window, stateful, center_of_interest, task, full_sequence, return_sequences):
    """
    Callable method to start the training of the model

    Parameters:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -model: name of the model to train, either CNN or LSTM
    -segmentation_value: window segmentation value in second
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
    -return_sequences (--return_sequences): true to return the state of each data point in the full sequence for the LSTM model
    """

    segmentation_value = float(segmentation_value)
    downsampling_value = float(downsampling_value)
    te_accuracy, te_f1 = evaluate_model(analysis_directory, model, segmentation_value, downsampling_value, epochs, data_balancing, log_time, batch_size, patience, standard_scale, sliding_window, stateful, center_of_interest, task, full_sequence, return_sequences)
    print('test accuracy:', te_accuracy*100, '%')
    print('test f1:', te_f1*100, '%')
    print('-----')
