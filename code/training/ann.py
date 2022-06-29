import wandb
from wandb.keras import WandbCallback

import os
import sys
import statistics
import numpy as np

from math import sqrt
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from keras import Sequential
from keras.layers import Conv1D, LSTM, Dropout, Dense, Activation, BatchNormalization, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
                sliding_window, 
                center_of_interest, 
                task, 
                wandb_log):
    """
    Method used to create and evaluate a deep learning model on data, either MLP, FCN, ResNet or LSTM

    Parameters:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -model_name: name of the model to train, either MLP, FCN, ResNet or LSTM
    -segmentation_value: window segmentation value in second
    -downsampling_value: signal downsampling value in second
    -epochs: number of epochs to train the model
    -data_balancing: true if balanced data is needed, false otherwise
    -log_time: save the time when the computation starts for the directory name
    -batch_size: batch size for training
    -patience: number of epochs without learning before early stop the training
    -sliding_window: true to use sliding window with a small center portion of interest
    -center_of_interest: center of interest size in seconds for the sliding window
    -task: corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep
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
        mode='min',
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
                                                  sliding_window=sliding_window, 
                                                  center_of_interest=center_of_interest, 
                                                  task=task).create_dataset()

    model = Sequential()
    n_timesteps, n_features, n_outputs, validation_split, verbose = X_train.shape[1], 1, 2, 0.1, 1
    
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
    elif model_name == 'fcn':
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
        lstm_units = max(24, int(2/3 * (int(segmentation_value * (1 / downsampling_value)) * n_outputs)))   # https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
        model.add(LSTM(units=lstm_units, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.2))
        model.add(Dense(lstm_units//2, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(lstm_units//4, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=n_outputs))
        model.add(Activation('softmax'))

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
    classes = np.argmax(predictions, axis=1)

    total_y_test = []
    for item in y_test:
        total_y_test.append(np.argmax(item))
    y_test = total_y_test

    # 95 % confidence interval computation
    interval = 1.96 * sqrt((te_accuracy * (1 - te_accuracy)) / len(X_test))

    # confusion matrix np array creation based on the prediction made by the model on the test data
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
    model_info_file.write(f'Sliding window = {int(sliding_window)} \n')
    model_info_file.write(f'Center of interest = {center_of_interest} \n')
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
