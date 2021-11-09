import os
import sys
import math
import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import copy
from scipy import stats
from keras import backend as K
from datetime import datetime as dt


def convert_string_to_time(start_time, start_date):
    temp_list = list(start_time)
    temp_list[len(temp_list) - 4] = '.'
    start_time = ''.join(temp_list)
    date_time = start_date + ' ' + start_time
    form = '%d.%m.%y %H:%M:%S.%f'
    return dt.strptime(date_time, form)


def extract_data_from_line(line):
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
    out = []
    start_datetime = convert_string_to_time(start_time, start_date)
    for i in range(len(times)):
        temp_time = start_datetime + datetime.timedelta(0, times[i])
        out.append(temp_time)
    return out


def datetime_conversion(times, start):
    out = []
    for i in range(len(times)):
        temp_time = start + datetime.timedelta(0, times[i])
        out.append(temp_time)
    return out


def hours_conversion(num_hours):
    frac, whole = math.modf(num_hours)
    return f'{str(int(whole))}h{str(int(frac*60))}min'


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def check_nan(values):
    out = []
    for arr in values:
        if np.isnan(np.sum(arr)):
            out.append(False)
        else:
            out.append(True)
    return pd.Series(out)


def encoder(df):
    for i in range(len(df['label'])):
        if df.at[i, 'label'] == 'Out of Range':
            df.at[i, 'label'] = 0
        else:
            df.at[i, 'label'] = 1


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
    out = {}
    for label in arr:
        idx = np.argmax(label)
        if idx in out.keys():
            out[idx] += 1
        else:
            out[idx] = 1
    return out


def remove_outlier(df, threshold):
    df['z_score'] = stats.zscore(df.data)
    df.drop(df.loc[df['z_score'].abs() > threshold].index.tolist(), inplace=True)
    df = df.drop(labels='z_score', axis=1)
    return df


def analysis_cutting(classes, analysis_start, analysis_end, time_step, threshold):
    classes_copy = copy(classes)
    analysis_start_copy = copy(analysis_start)
    analysis_end_copy = copy(analysis_end)
    time_step_copy = copy(time_step)
    classes_df = pd.DataFrame(columns=['start_time', 'end_time', 'label'])
    curr_time = analysis_start
    for label in classes:
        temp = [curr_time, curr_time + datetime.timedelta(minutes=time_step), label[0]]
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
        if row.label != 0:
            valid_count += 1
    valid_hours = (valid_count*time_step) / 60
    if len(classes_df) > 0:
        valid_rate = valid_count / len(classes_df)
    else:
        valid_rate = 0

    if threshold != 0:
        if not is_valid(valid_hours, valid_rate):
            print('new start analysis time:', new_start_time)
            print('new end analysis time:', new_end_time)
            print('analysis duration:', (new_end_time-new_start_time))
            print('valid time in new bounds:', hours_conversion(valid_hours))
            print('valid rate in new bounds:', valid_rate*100)
            print("not enough valid signal (< 4h) in first selected bounds, let's try with a wider tolerance")
            return analysis_cutting(classes_copy, analysis_start_copy, analysis_end_copy, time_step_copy, threshold=0)
    return valid_hours, valid_rate, new_start_time, new_end_time


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def binary_y_format_revert(y):
    out = []
    for arr in y:
        out.append(np.argmax(arr))
    return np.array(out)


def num_of_correct_pred(y_true, y_pred):
    count = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            count += 1
    return count


def is_valid(valid_hours, valid_rate):
    if valid_hours > 6:
        return True
    elif valid_hours > 4 and valid_rate > 0.75:
        return True
    return False
