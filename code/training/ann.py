import wandb
from wandb.keras import WandbCallback

import os
import sys
import time
import itertools
import statistics
import numpy as np

from math import sqrt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from tqdm import tqdm
from matplotlib import pyplot as plt
from keras import Sequential
from keras.layers import Conv1D, LSTM, Dropout, MaxPooling1D, Flatten, Dense, Activation, BatchNormalization, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')

from data_loader import DataLoader
from util import plot_confusion_matrix, TimingCallback, f1


def train_model(analysis_directory, 
                model_name, 
                segmentation_value, 
                downsampling_value, 
                epochs, 
                data_balancing, 
                log_time, 
                batch_size, 
                patience, 
                standard_scale, 
                sliding_window, 
                stateful, 
                center_of_interest, 
                task, 
                full_sequence, 
                return_sequences, 
                wandb_log):
    """
    Method used to create and evaluate a deep learning model on data, either MLP, CNN, ResNet or LSTM

    Parameters:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -model_name: name of the model to train, either MLP, CNN, ResNet or LSTM
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
    -task: corresponding task number, so far: task = 1 for valid/invalid, task = 2 for awake/sleep and task = 3 for belts analysis
    -full_sequence: true to feed the entire sequence without dividing it into multiple windows
    -return_sequences: true to return the state of each data point in the full sequence for the LSTM model
    -wandb_log: true to log the training results on weights and biases
    """

    if wandb_log:
        wandb.init(project="nomics-AI", entity="nomics")

        wandb.config = {
            "epochs": epochs,
            "batch_size": batch_size
        }

    segmentation_value = float(segmentation_value)
    downsampling_value = float(downsampling_value)

    # directory creation
    save_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task{task}/{model_name}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_dir += f'/{log_time}'
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

    X_train, y_train, X_test, y_test = DataLoader(analysis_directory=analysis_directory, 
                                                  model_name = model_name,
                                                  segmentation_value=segmentation_value, 
                                                  downsampling_value=downsampling_value, 
                                                  data_balancing=data_balancing, 
                                                  log_time=log_time, 
                                                  standard_scale=standard_scale, 
                                                  sliding_window=sliding_window, 
                                                  stateful=stateful, 
                                                  center_of_interest=center_of_interest, 
                                                  task=task, 
                                                  full_sequence=full_sequence, 
                                                  return_sequences=return_sequences).create_dataset()

    model = Sequential()
    # Stateless model
    if not stateful:
        if not full_sequence:
            n_timesteps, n_features, n_outputs, validation_split, verbose = X_train.shape[1], 1, 2, 0.1, 1
        else:
            n_timesteps, n_features, n_outputs, validation_split, verbose = None, 1, 2, 0.1, 1
        
        if model_name == 'mlp':
            model.add(Dense(500, input_dim=1, activation='relu'))
            model.add(Dropout(0.1))
            model.add(Dense(500, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(500, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(500, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(n_outputs, activation='sigmoid'))
        elif model_name == 'cnn':
            model.add(Conv1D(filters=128, kernel_size=8, strides=1, activation='relu', input_shape=(n_timesteps, n_features)))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=256, kernel_size=5, strides=1, activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu'))
            model.add(BatchNormalization())

            model.add(GlobalAveragePooling1D())

            model.add(Dense(n_outputs, activation='softmax'))
        elif model_name == 'resnet':
            model.add(Conv1D(filters=64, kernel_size=8, strides=1, activation='relu', input_shape=(n_timesteps, n_features)))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=64, kernel_size=8, strides=1, activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=64, kernel_size=8, strides=1, activation='relu'))
            model.add(BatchNormalization())

            model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='relu'))
            model.add(BatchNormalization())

            model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu'))
            model.add(BatchNormalization())

            model.add(GlobalAveragePooling1D())

            model.add(Dense(n_outputs, activation='softmax'))
        elif model_name == 'lstm':
            stf = False
            lstm_units = max(24, int(2/3 * (int(segmentation_value * (1 / downsampling_value)) * n_outputs)))   # https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
            if task == 1:
                model.add(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=stf, input_shape=(n_timesteps, n_features)))   # lstm layer -- 1
            else:
                # model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=stf), input_shape=(n_timesteps, n_features)))   # lstm layer -- 1
                model.add(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=stf, input_shape=(n_timesteps, n_features)))   # lstm layer -- 1
            if return_sequences:
                if task == 1:
                    model.add(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=stf))   # lstm layer -- 1'
                else:
                    model.add(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=stf))   # lstm layer -- 1'
                    # model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=stf)))   # lstm layer -- 1'
            model.add(Dropout(0.2))   # dropout -- 2
            model.add(Dense(lstm_units//2, activation='relu'))   # fully connected layer -- 3
            model.add(Dropout(0.2))   # dropout -- 4
            model.add(Dense(lstm_units//4, activation='relu'))   # fully connected layer -- 5
            model.add(Dropout(0.2))   # dropout -- 6
            model.add(Dense(units=n_outputs))   # fully connected layer -- 7
            model.add(Activation('softmax'))   # activation -- 8

        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy', f1])

        print(model.summary())

        time_callback = TimingCallback()

        if wandb_log:
            callbacks = [time_callback, model_checkpoint_callback, early_stop, WandbCallback()]
        else:
            callbacks = [time_callback, model_checkpoint_callback, early_stop]

        history = model.fit(X_train, 
                            y_train, 
                            validation_split=validation_split, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            verbose=verbose, 
                            callbacks=callbacks)

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
        n_timesteps, n_features, n_outputs, validation_split, return_sequences = max_length, 1, 2, 0.30, True
        lstm_units = max(8, int(2/3 * (n_timesteps * n_outputs)))   # https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
    
        model.add(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=True, batch_input_shape=(batch_size, n_timesteps, n_features)))   # lstm layer -- 1
        model.add(LSTM(units=lstm_units, return_sequences=return_sequences, stateful=True))   # lstm layer -- 1'
        model.add(Dropout(0.2))   # dropout -- 2
        model.add(Dense(lstm_units//2, activation='relu'))   # fully connected layer -- 3
        model.add(Dropout(0.2))   # dropout -- 4
        model.add(Dense(lstm_units//4, activation='relu'))   # fully connected layer -- 5
        model.add(Dropout(0.2))   # dropout -- 6
        model.add(Dense(units=n_outputs))   # fully connected layer -- 7
        model.add(Activation('softmax'))   # activation -- 8

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])

        print(model.summary())

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, random_state=42)
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split)

        X_tr = []
        y_tr = []
        for idx in range(len(X_train)):
            temp_X_tr = [np.reshape(X_train[idx][i:i + max_length], (n_timesteps, n_features)) for i in range(0, len(X_train[idx])-round(((len(X_train[idx])/(max_length*batch_size))%1)*(max_length*batch_size))-1, max_length)]
            temp_y_tr = [np.array(y_train[idx][i:i + max_length]) for i in range(0, len(y_train[idx])-round(((len(y_train[idx])/(max_length*batch_size))%1)*(max_length*batch_size))-1, max_length)]
            X_tr.append(np.array(temp_X_tr))
            y_tr.append(np.array(to_categorical(temp_y_tr, num_classes=2)))

        X_v = []
        y_v = []
        for idx in range(len(X_val)):
            temp_X_v = [np.reshape(X_val[idx][i:i + max_length], (n_timesteps, n_features)) for i in range(0, len(X_val[idx])-round(((len(X_val[idx])/(max_length*batch_size))%1)*(max_length*batch_size))-1, max_length)]
            temp_y_v = [np.array(y_val[idx][i:i + max_length]) for i in range(0, len(y_val[idx])-round(((len(y_val[idx])/(max_length*batch_size))%1)*(max_length*batch_size))-1, max_length)]
            X_v.append(np.array(temp_X_v))
            y_v.append(np.array(to_categorical(temp_y_v, num_classes=2)))
                
        X_te = []
        y_te = []
        for idx in range(len(X_test)):
            temp_X_te = [np.reshape(X_test[idx][i:i + max_length], (n_timesteps, n_features)) for i in range(0, len(X_test[idx])-round(((len(X_test[idx])/(max_length*batch_size))%1)*(max_length*batch_size))-1, max_length)]
            temp_y_te = [np.array(y_test[idx][i:i + max_length]) for i in range(0, len(y_test[idx])-round(((len(y_test[idx])/(max_length*batch_size))%1)*(max_length*batch_size))-1, max_length)]
            X_te.append(np.array(temp_X_te))
            y_te.append(np.array(to_categorical(temp_y_te, num_classes=2)))
        
        computation_time_history = []
        training_loss_history = []
        training_accuracy_history = []
        training_f1_history = []
        validation_loss_history = []
        validation_accuracy_history = []
        validation_f1_history = []

        best_val_loss = sys.maxsize
        patience_count = 0

        # train model
        for i in range(epochs):
            print('----- EPOCH #{} -----'.format(i+1))
            temp_training_loss = []
            temp_training_accuracy = []
            temp_training_f1 = []
            print('--> TRAIN...')
            start = time.time()
            for j in tqdm(range(len(X_tr))):
                if len(y_tr[j].shape) > 2:
                    y_integers = list(itertools.chain.from_iterable(np.argmax(y_tr[j], axis=2)))
                    sample_weights = compute_sample_weight('balanced', y_integers)
                    sample_weights = np.reshape(sample_weights, (len(X_tr[j]), max_length))
                    history = model.fit(X_tr[j], 
                                        y_tr[j], 
                                        epochs=1, 
                                        batch_size=batch_size, 
                                        verbose=0, 
                                        shuffle=False, 
                                        sample_weight=sample_weights)
                    model.reset_states()
                    temp_training_loss.append(history.history['loss'][-1])
                    temp_training_accuracy.append(history.history['accuracy'][-1])
                    temp_training_f1.append(history.history['f1'][-1])
                else:
                    print(f'num {j} with shape {y_tr[j].shape} not accepted')
                    continue

            print('-----')

            temp_validation_loss = []
            temp_validation_accuracy = []
            temp_validation_f1 = []
            print('--> VAL...')
            for j in tqdm(range(len(X_v))):
                if len(y_v[j].shape) > 2:
                    dic = model.evaluate(X_v[j], 
                                         y_v[j], 
                                         batch_size=batch_size, 
                                         verbose=0, 
                                         return_dict=True)
                    model.reset_states()
                    temp_validation_loss.append(dic['loss'])
                    temp_validation_accuracy.append(dic['accuracy'])
                    temp_validation_f1.append(dic['f1'])
                else:
                    print(f'num {j} with shape {y_v[j].shape} not accepted')
                    continue
            end = time.time()

            computation_time_history.append(end-start)

            tr_loss = np.mean(temp_training_loss)
            tr_acc = np.mean(temp_training_accuracy)
            tr_f1 = np.mean(temp_training_f1)
            val_loss = np.mean(temp_validation_loss)
            val_acc = np.mean(temp_validation_accuracy)
            val_f1 = np.mean(temp_validation_f1)

            training_loss_history.append(tr_loss)
            training_accuracy_history.append(tr_acc)
            training_f1_history.append(tr_f1)
            validation_loss_history.append(val_loss)
            validation_accuracy_history.append(val_acc)
            validation_f1_history.append(val_f1)

            print()
            print(f'train loss = {round(tr_loss, 3)}')
            print(f'train acc = {round(tr_acc, 3)}')
            print(f'train f1 = {round(tr_f1, 3)}')
            print()
            print(f'val loss = {round(val_loss, 3)}')
            print(f'val acc = {round(val_acc, 3)}')
            print(f'val f1 = {round(val_f1, 3)}')
            print()
            print(f'--> timer = {round(end-start, 2)} sec, {round((end-start)/60, 2)} min')
            print()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(f'{save_dir}/best/model-best.h5')
                if wandb_log:
                    model.save(os.path.join(wandb.run.dir, "model-best.h5"))
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                break

            if wandb_log:
                wandb.log({'loss': tr_loss, 
                           'accuracy': tr_acc, 
                           'f1': tr_f1, 
                           'val_loss': val_loss, 
                           'val_accuracy': val_acc, 
                           'val_f1': val_f1})

        print('----- TRAIN DONE -----')

        temp_test_loss = []
        temp_test_accuracy = []
        temp_test_f1 = []
        # evaluate model
        print('--> TEST...')
        for i in tqdm(range(len(X_te))):
            dic = model.evaluate(X_te[i], 
                                 y_te[i], 
                                 batch_size=batch_size, 
                                 verbose=0, 
                                 return_dict=True)
            model.reset_states()

            temp_test_loss.append(dic['loss'])
            temp_test_accuracy.append(dic['accuracy'])
            temp_test_f1.append(dic['f1'])

        te_loss = np.mean(temp_test_loss)
        te_accuracy = np.mean(temp_test_accuracy)
        te_f1 = np.mean(temp_test_f1)

        print(f'test loss = {round(te_loss, 3)}')
        print(f'test acc = {round(te_accuracy, 3)}')
        print(f'test f1 = {round(te_f1, 3)}')

        classes = []
        y_test = []
        # predict model
        print('--> PREDICT...')
        for i in tqdm(range(len(X_te))):
            predictions = model.predict(X_te[i], 
                                        batch_size=batch_size)
            temp_classes = []
            for c in np.argmax(predictions, axis=2):
                temp_classes.extend(c)
            classes.append(temp_classes)
            y_test.append(list(itertools.chain.from_iterable(np.argmax(y_te[i], axis=2))))

    # 95 % confidence interval computation
    interval = 1.96 * sqrt((te_accuracy * (1 - te_accuracy)) / len(X_test))

    # confusion matrix np array creation based on the prediction made by the model on the test data
    if not return_sequences:
        cm = confusion_matrix(y_true=y_test, y_pred=classes)
    elif stateful:
        cm = confusion_matrix(y_true=list(itertools.chain.from_iterable(y_test)), y_pred=list(itertools.chain.from_iterable(classes)))
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
    if task == 1 or task == 3:
        labels = ['Invalid', 'Valid']
    elif task == 2:
        labels = ['Awake', 'Sleep']
    
    cm_plt = plot_confusion_matrix(cm=cm, classes=labels, title='Confusion Matrix', normalize=True)

    cm_plt.savefig(f'{save_dir}/cm_plt.png', bbox_inches='tight')
    if wandb_log:
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
    if wandb_log:
        model.save(os.path.join(wandb.run.dir, "model-last.h5"))

    print('test accuracy:', te_accuracy*100, '%')
    print('test f1:', te_f1*100, '%')
    print('-----')
