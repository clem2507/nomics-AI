import os
import sys
import math
import time
import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import copy
from tqdm import tqdm
from scipy import stats
from keras import backend as K
from keras.callbacks import Callback
from datetime import datetime as dt
from sklearn.model_selection import train_test_split


def convert_string_to_time(time, date):
    """
    Method used to convert string time to time dtype

    Parameters:

    -time: input time to be converted
    -date: input date to be converted

    Returns:

    -out_time: output time in correct format
    """

    temp_list = list(time)
    temp_list[len(temp_list) - 4] = '.'
    out_time = ''.join(temp_list)
    out_time = date + ' ' + out_time
    form = '%d.%m.%y %H:%M:%S.%f'
    return dt.strptime(out_time, form)


def extract_data_from_line(line):
    """
    Method used to extract the essential information from a mk3 label line

    Parameters:

    -line: mk3 labelling line

    Returns:

    -out: list with the important information to keep from the line
    """

    out = []
    temp = line.split(';')
    # start date and time
    out.append(convert_string_to_time(temp[0], temp[1]))
    # end date and time
    out.append(convert_string_to_time(temp[2], temp[3]))
    # zone type
    out.append(temp[len(temp) - 1][:len(temp[len(temp) - 1]) - 1])
    return out


def string_datetime_conversion(times, start_time, start_date):
    """
    Function that converts a list of string times to a list of date time dtype

    Parameters:

    -times: list containing the time in string format
    -start_time: start time of the first recorded data point
    -start_date: start date of the first recorded data point

    Returns:

    -out: list with date time values
    """

    out = []
    start_datetime = convert_string_to_time(start_time, start_date)
    for i in range(len(times)):
        temp_time = start_datetime + datetime.timedelta(0, times[i])
        out.append(temp_time)
    return out


def datetime_conversion(times, start):
    """
    Function that converts a list of string times to a list of date time dtype

    Parameters:

    -times: list containing the time in string format
    -start: start date time of the first recorded data point

    Returns:

    -out: list with date time values
    """

    out = []
    for i in range(len(times)):
        temp_time = start + datetime.timedelta(0, times[i])
        out.append(temp_time)
    return out


def hours_conversion(num_hours):
    """
    Method used to convert a number of hours to hours with minut format
    ex: 1.8h -> 1h48min

    Parameters:

    -num_hours: total number of hours

    Returns:

    total number of hours with minutes
    """

    frac, whole = math.modf(num_hours)
    return f'{str(int(whole))}h{str(int(frac * 60))}min'


def block_print():
    """
    Function that stops print statement until enable_print() is called back
    """

    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """
    Function that enables print statement again
    """

    sys.stdout = sys.__stdout__


def check_nan(values):
    """
    Method used to find nan values in dataframes series and delete these rows

    Parameters:

    -values: series with the time series data point

    Returns:

    -out: new series without nan values
    """

    out = []
    for arr in values:
        if np.isnan(np.sum(arr)):
            out.append(False)
        else:
            out.append(True)
    return pd.Series(out)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Confustion matrix plotter function

    Parameters:

    -cm: confusion matrix numpy array
    -classes: classes names
    -normalize: true if normalized values are desired inside the confusion matrix, otherwise false
    -title: plot title
    -cmap: color of the map

    Returns:

    -plt: plot containing the confusion matrix
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def occurrences_counter(arr):
    """
    Function that counts the total number of different classes

    Parameters:

    -arr: array with classes values

    Returns:

    -out: dictionary with the classes occurence values
    """

    out = {}
    for label in arr:
        idx = np.argmax(label)
        if idx in out.keys():
            out[idx] += 1
        else:
            out[idx] = 1
    return out


def analysis_cutting(classes, analysis_start, analysis_end, time_step, threshold):
    """
    Method that computes new analysis bounds based on the different segmented windows label

    Parameters:

    -classes: analysis predicted classes
    -analysis_start: analysis start time
    -analysis_end: analysis end time
    -time_step: segmentation time step in minutes
    -threshold: minimum threshold validity of the signal before stopping the reduction of the bounds

    Returns:

    -valid_hours: total number of valid hours in the new selected bounds
    -valid_rate: valid hours rate inside the new selected bounds
    -new_start_time: time of the new starting bound
    -new_end_time: time of the new ending bound
    """

    classes_copy = copy(classes)
    analysis_start_copy = copy(analysis_start)
    analysis_end_copy = copy(analysis_end)
    time_step_copy = copy(time_step)
    classes_df = pd.DataFrame(columns=['start_time', 'end_time', 'label', 'probability'])
    curr_time = analysis_start
    for label in classes:
        temp = [curr_time, curr_time + datetime.timedelta(minutes=time_step), label[0], label[1]]
        classes_df = classes_df.append(pd.Series(temp, index=classes_df.columns), ignore_index=True)
        curr_time += datetime.timedelta(minutes=time_step)

    flag = False
    new_start_time = None
    new_end_time = None
    while flag is False:
        if new_start_time is None or flag is False:
            for idx, row in classes_df.iterrows():
                if row.label == 1:
                    new_start_time = row.start_time
                    break
                else:
                    classes_df.drop(index=idx, inplace=True)

        if new_end_time is None or flag is False:
            for idx, row in classes_df.iloc[::-1].iterrows():
                if row.label == 1:
                    new_end_time = row.end_time
                    break
                else:
                    classes_df.drop(index=idx, inplace=True)

        if threshold == 0:
            break

        if 0 in classes_df.label.values and 1 in classes_df.label.values:
            if (classes_df.label.value_counts()[1] / len(classes_df)) > threshold:
                flag = True
            else:
                first_half = classes_df.iloc[:int(len(classes_df) / 2), :]
                second_half = classes_df.iloc[int(len(classes_df) / 2):, :]
                if 0 in first_half.label.values and 0 in second_half.label.values:
                    if first_half.label.value_counts()[0] > second_half.label.value_counts()[0]:
                        classes_df = classes_df[1:]
                    else:
                        classes_df = classes_df[:-1]
                if 0 in first_half.label.values and 0 not in second_half.label.values:
                    classes_df = classes_df[1:]
                if 0 not in first_half.label.values and 0 in second_half.label.values:
                    classes_df = classes_df[:-1]
        else:
            flag = True

        if len(classes_df) < 1:
            flag = True
            new_start_time = analysis_start
            new_end_time = analysis_end

    valid_count = 0
    for idx, row in classes_df.iterrows():
        if row.label == 1:
            valid_count += 1
    valid_hours = (valid_count * time_step) / 60
    if len(classes_df) > 0:
        valid_rate = valid_count / len(classes_df)
    else:
        valid_rate = 0

    if threshold != 0:
        if not is_valid(analysis_end - analysis_start, valid_hours):
            print('new start analysis time:', new_start_time)
            print('new end analysis time:', new_end_time)
            print('analysis duration:', (new_end_time - new_start_time))
            print('valid time in new bounds:', hours_conversion(valid_hours))
            print('valid rate in new bounds:', round((valid_rate * 100), 2), '%')
            print(f"not enough valid signal (< 4h) in first selected bounds with threshold = {threshold}, let's try with a wider tolerance")
            if threshold == 0.98:
                return analysis_cutting(classes_copy, analysis_start_copy, analysis_end_copy, time_step_copy, threshold=0.8)
            if threshold == 0.8:
                return analysis_cutting(classes_copy, analysis_start_copy, analysis_end_copy, time_step_copy, threshold=0.5)
            if threshold == 0.5:
                return analysis_cutting(classes_copy, analysis_start_copy, analysis_end_copy, time_step_copy, threshold=0)
    new_bounds_classes = []
    for idx, row in classes_df.iterrows():
        new_bounds_classes.append((row.label, row.probability)) 
    return new_bounds_classes, valid_hours, valid_rate, new_start_time, new_end_time


def recall_m(y_true, y_pred):
    """
    Method that computes the recall based on predicted and true labels

    Parameters:

    -y_true: true labels list of x instances
    -y_pred: predicted labels list of x instances

    Returns:

    -recall: measurement of the recall
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
    Method that computes the precision based on predicted and true labels

    Parameters:

    -y_true: true labels list of x instances
    -y_pred: predicted labels list of x instances

    Returns:

    -precision: measurement of the precision
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """
    Method that computes the f1 score based on predicted and true labels

    Parameters:

    -y_true: true labels list of x instances
    -y_pred: predicted labels list of x instances

    Returns:

    -recall: measurement of the f1 score
    """

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def num_of_correct_pred(y_true, y_pred):
    """
    Method that computes the total number of correct predictions made by the classifier

    Parameters:

    -y_true: true labels list of x instances
    -y_pred: predicted labels list of x instances

    Returns:

    -count: the total count of correct predictions
    """

    count = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            count += 1
    return count


def is_valid(total_hours, valid_hours):
    """
    Function that takes the decision about the validity of an analysis based on its total number of valid hours and rate
    
    If more than 6 hours of validity available, analysis is directy valid
    If the validity time is between 4 and 6 hours, it must have at least 75% within the limits
    If less than 4 hours, directly classified as invalid

    Parameters:

    -valid_hours: total number of valid hours
    -valid_rate: total rate of valid hours

    Returns:

    True if the entire analysis is valid, false otherwise
    """
    if (total_hours.total_seconds() / 3600.0) > 6 and valid_hours > 4:
        return True
    return False


def signal_quality(classes):
    """
    Method used to give an estimate of the valid quality of the signal based on the output probabilities of the model

    Parameters:

    -classes: list with the output predicted classes with their probabilities

    Returns:

    -out: signal quality estimation (in %)
    """
    score = 0
    if len(classes) == 0:
        return score
    for c in classes:
        if c[0] == 0:
            score += 1-c[1]
        elif c[0] == 1:
            score += c[1]
    out = score/len(classes)
    if out < 0:
        out = 0
    return out


def stateful_fit(model, X_train, y_train, epochs):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print('Train...')
    for epoch in range(epochs):
        print('epoch #{}'.format(epoch+1))
        print('--->')
        mean_tr_acc = []
        mean_tr_loss = []
        for i in tqdm(range(len(X_train))):
            for j in range(len(X_train[i])):
                for k in range(len(X_train[i][j])):
                    tr_loss, tr_acc, *r = model.train_on_batch(np.expand_dims(np.expand_dims([X_train[i][j][k]], axis=1), axis=1), np.array([y_train[i][j][k]]))
                    mean_tr_acc.append(tr_acc)
                    mean_tr_loss.append(tr_loss)
            model.reset_states()
        print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
        print('loss training = {}'.format(np.mean(mean_tr_loss)))

        print('<---')
        mean_te_acc = []
        mean_te_loss = []
        for i in tqdm(range(len(X_val))):
            for j in range(len(X_val[i])):
                for k in range(len(X_val[i][j])):
                    te_loss, te_acc, *r = model.test_on_batch(np.expand_dims(np.expand_dims([X_val[i][j][k]], axis=1), axis=1), np.array([y_val[i][j][k]]))
                    mean_te_acc.append(te_acc)
                    mean_te_loss.append(te_loss)
            model.reset_states()

            # for j in range(len(X_val[i])):
            #     y_pred, *r = model.predict_on_batch(np.expand_dims(np.expand_dims(X_val[i][j], axis=1), axis=1))
            # model.reset_states()

        print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
        print('loss testing = {}'.format(np.mean(mean_te_loss)))
    
    return model


class TimingCallback(Callback):
    """
    Class used to save the training computation time after each epoch
    """

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.logs = []

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.logs.append(time.time() - self.start_time)

class ResetStatesCallback(Callback):
    def __init__(self, max_len):
        self.counter = 0
        self.max_len = max_len

    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.max_len == 0:
            self.model.reset_states()
        self.counter += 1