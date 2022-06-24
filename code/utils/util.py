import os
import sys
import math
import time
import datetime
import itertools
import numpy as np
import matplotlib.pyplot as plt

from copy import copy

from keras import backend as K
from keras.callbacks import Callback
from datetime import datetime as dt


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
    # signal number
    if len(temp[len(temp) - 2]) == 1:
        out.append(int(temp[len(temp) - 2]))
    else:
        out.append(-1)
    # zone type
    out.append((temp[len(temp) - 1]).strip())
    return [out]


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
        # out.append(pd.Timestamp(temp_time))
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
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def analysis_cutting(df, analysis_start, analysis_end, downsampling_value, threshold):
    """
    Method that computes new analysis bounds based on the different segmented windows label

    Parameters:

    -df: analysis predicted df
    -analysis_start: analysis start time
    -analysis_end: analysis end time
    -downsampling_value: signal downsampling value in second
    -threshold: minimum threshold validity of the signal before stopping the reduction of the bounds

    Returns:

    -valid_hours: total number of valid hours in the new selected bounds
    -valid_rate: valid hours rate inside the new selected bounds
    -new_start_time: time of the new starting bound
    -new_end_time: time of the new ending bound
    """

    df_copy = copy(df)
    analysis_start_copy = copy(analysis_start)
    analysis_end_copy = copy(analysis_end)
    downsampling_value_copy = copy(downsampling_value)

    flag = False
    new_start_time = None
    new_end_time = None
    while flag is False:
        if new_start_time is None or flag is False:
            for idx, row in df.iterrows():
                if row.label == 1:
                    new_start_time = row.start
                    break
                else:
                    df = df.drop(index=idx)

        if new_end_time is None or flag is False:
            for idx, row in df.iloc[::-1].iterrows():
                if row.label == 1:
                    new_end_time = row.end
                    break
                else:
                    df = df.drop(index=idx)

        if threshold == 0:
            break

        if (0 in df.label.values and 1 in df.label.values) or (1 in df.label.values and 2 in df.label.values):
            if (sum(df[df['label']==1].data_num.tolist()) / sum(df.data_num.tolist())) > threshold:
                flag = True
            else:
                first_half = df.iloc[:int(sum(df.data_num.tolist()) / 2), :]
                second_half = df.iloc[int(sum(df.data_num.tolist()) / 2):, :]
                if (0 in first_half.label.values and 0 in second_half.label.values):
                    if first_half.label.value_counts()[0] > second_half.label.value_counts()[0]:
                        df = df[1:]
                    else:
                        df = df[:-1]
                if (2 in first_half.label.values and 2 in second_half.label.values):
                    if first_half.label.value_counts()[2] > second_half.label.value_counts()[2]:
                        df = df[1:]
                    else:
                        df = df[:-1]
                if (0 in first_half.label.values and 0 not in second_half.label.values) or (2 in first_half.label.values and 2 not in second_half.label.values):
                    df = df[1:]
                if (0 not in first_half.label.values and 0 in second_half.label.values) or (2 not in first_half.label.values and 2 in second_half.label.values):
                    df = df[:-1]
        else:
            flag = True

        if len(df) < 1:
            flag = True
            new_start_time = analysis_start
            new_end_time = analysis_end

    
    sleep_count = sum(df[df['label']==1].data_num.tolist())
    sleep_hours = (sleep_count * downsampling_value) / 3600
    sleep_rate = 0
    if len(df) > 0:
        sleep_rate = sleep_count / df['data_num'].sum()

    if threshold != 0:
        if not is_valid(sleep_hours, threshold=4):
            print('threshold:', threshold)
            print('new start analysis time:', new_start_time)
            print('new end analysis time:', new_end_time)
            print('analysis duration:', (new_end_time - new_start_time))
            print('sleep time in new bounds:', hours_conversion(sleep_hours))
            print('sleep rate in new bounds:', round((sleep_rate * 100), 2), '%')
            print(f"not enough valid signal (< 4h) in first selected bounds with threshold = {threshold}, let's try with a wider tolerance")
            print()
            if threshold == 0.95:
                return analysis_cutting(df_copy, analysis_start_copy, analysis_end_copy, downsampling_value_copy, threshold=0.8)
            if threshold == 0.8:
                return analysis_cutting(df_copy, analysis_start_copy, analysis_end_copy, downsampling_value_copy, threshold=0.5)
            if threshold == 0.5:
                return analysis_cutting(df_copy, analysis_start_copy, analysis_end_copy, downsampling_value_copy, threshold=0)
    return df_copy, sleep_hours, sleep_rate, new_start_time, new_end_time


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


def f1(y_true, y_pred):
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


def is_valid(sleep_hours, threshold=4):
    """
    Function that takes the decision about the validity of an analysis based on its total number of sleep hours

    Parameters:

    -sleep_hours: total hours of sleep

    Returns:

    True if the entire analysis is valid, false otherwise
    """

    if sleep_hours > threshold:
        return True
    return False


def signal_quality(df):
    """
    Method used to give an estimate of the valid quality of the signal based on the output probabilities of the model

    Parameters:

    -df: df with the output predicted classes with their probabilities

    Returns:

    -out: signal quality estimation (in %)
    """

    score = 0
    start = df.start.tolist()[0]
    end = df.end.tolist()[-1]
    time_interval = end - start
    if len(df) == 0:
        return score
    for idx, row in df.iterrows():
        if row.label == 1:
            score += row.proba*((row.end-row.start)/time_interval)
        else:
            score += (1-row.proba)*((row.end-row.start)/time_interval)
    if score < 0:
        score = 0
    return score


def get_value_in_line(line):
    """
    Method used to access the correct value in the info.txt file from models

    Parameters:

    -line: line to find the value in

    Returns:

    -out: the correct value
    """

    out = ''
    flag = False
    for i in range(len(line)):
        if len(out) > 0 and flag and line[i] == ' ':
            return out
        if flag:
            out += line[i]
        if line[i] == '=':
            flag = True
    return out


def occurrence_count(list, threshold=0.9):
    """
    Method used to check if there are enough classes of the same type in a given data window

    Parameters:

    -list: list corresponding to the data window

    Returns:

    -out: true if at least > threshold distribution of one label in this window 
    """

    dic = {}
    for item in list:
        if str(item) not in dic.keys():
            dic[str(item)] = 1
        else:
            dic[str(item)] += 1
    total = 0
    for k in dic.keys():
        total += dic[k]
    for k in dic.keys():
        if dic[k]/total > threshold:
            return True
    return False


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
