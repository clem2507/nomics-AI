import os
import time
import mne
import datetime
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import stats
from keras import backend as K
from datetime import datetime as dt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tensorflow.keras.models import load_model


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
    valid_hours = (valid_count*time_step) / 60
    valid_rate = (valid_count / len(classes_df)*100)
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


def analysis_classification(edf, model, num_class):

    start = time.time()
    if num_class == 2:
        # time_split in minutes
        time_split = 1
        # time_resampling in seconds
        time_resampling = 1
        epochs = 5
        model_path = os.path.dirname(os.path.abspath('run_script.py')) + f'/models/{model.lower()}/binomial/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs/saved_model'
    else:
        # time_split in minutes
        time_split = 1
        # time_resampling in seconds
        time_resampling = 1
        epochs = 10
        model_path = os.path.dirname(os.path.abspath('run_script.py')) + f'/models/{model.lower()}/multinomial/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs/saved_model'

    # model loader
    if model.lower() in ['cnn', 'lstm']:
        if os.path.exists(model_path):
            model = load_model(model_path, compile=True, custom_objects={'f1_m': f1_m})
        else:
            raise Exception('model path does not exist')
    else:
        raise Exception('model should either be CNN or LSTM, found something else')

    full_path = os.path.dirname(os.path.abspath('run_script.py')) + '/' + edf

    # edf file
    raw_data = mne.io.read_raw_edf(full_path)
    data, times = raw_data[:]
    times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])
    df_jawac = pd.DataFrame()
    df_jawac.insert(0, 'times', times)
    df_jawac.insert(1, 'data', data[0])
    df_jawac = df_jawac.resample(str(time_resampling) + 'S', on='times').median()['data'].to_frame(name='data')
    df_jawac.reset_index(inplace=True)

    size = int((60 / time_resampling) * time_split)
    X_test_seq = np.array([df_jawac.loc[i:i + size - 1, :].data for i in range(0, len(df_jawac), size)], dtype=object)

    X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, padding='post', dtype='float64')

    X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))

    df_jawac.set_index('times', inplace=True)

    predictions = model.predict(X_test_seq_pad)
    classes = []
    for item in predictions:
        idx = np.argmax(item)
        classes.append((idx, item[idx]))

    valid_total = 0
    invalid_total = 0
    for label in classes:
        if label[0] == 1:
            valid_total += 1
        if label[0] == 0:
            invalid_total += 1
    valid_mean = valid_total / len(classes)
    invalid_mean = invalid_total / len(classes)

    print('--------')

    print('valid percentage:', (valid_mean * 100), '%')
    print('invalid percentage:', (invalid_mean * 100), '%')

    print('--------')

    print('valid hours:', ((valid_total * time_split) / 60), 'h')
    print('invalid hours:', ((invalid_total * time_split) / 60), 'h')

    print('--------')

    valid_hours, valid_rate, new_start, new_end = analysis_cutting(classes, df_jawac.index[0], df_jawac.index[-1], time_split, threshold=0.8)

    print('new start analysis time:', new_start)
    print('new end analysis time:', new_end)
    duration = (new_end - new_start)
    if new_start is not None and new_end is not None:
        print('analysis duration:', duration)
    else:
        print('analysis duration: Unknown')
    print('valid hours in new bounds:', valid_hours, 'h')
    print('valid rate in new bounds:', valid_rate, '%')

    print('--------')

    # graph
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    # ax.plot(df_jawac.resample(str(graph_time_resampling)+'S').median()['data'].to_frame(name='data').index.tolist(), df_jawac.resample(str(graph_time_resampling)+'S').median()['data'].to_frame(name='data').data.tolist())
    ax.plot(df_jawac.index.tolist(), df_jawac.data.tolist())
    ax.axhline(y=0, color='r', linewidth=1)
    if new_start is not None and new_end is not None:
        ax.axvline(x=new_start, color='k', linewidth=2, linestyle='--')
        ax.axvline(x=new_end, color='k', linewidth=2, linestyle='--')
    title = ''
    for c in reversed(edf[:-4]):
        if c != '/':
            title += c
        else:
            break
    title = title[::-1]
    ax.set(xlabel='time', ylabel='opening (mm)', title=f'Jawac Signal - {title}')
    ax.grid()

    curr_time = raw_data.__dict__['info']['meas_date']
    for label in classes:
        if label[0] == 0:
            # plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='r', alpha=0.3*label[1])
            plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='r', alpha=0.25)
        elif label[0] == 1:
            # plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='g', alpha=0.3*label[1])
            plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='g', alpha=0.25)
        elif label[0] == 2:
            # plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='b', alpha=0.3*label[1])
            plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='b', alpha=0.25)
        curr_time += datetime.timedelta(minutes=time_split)

    if num_class == 2:
        legend_elements = [Patch(facecolor='r', edgecolor='w', label='invalid area', alpha=0.2),
                           Patch(facecolor='g', edgecolor='w', label='valid area', alpha=0.2),
                           Line2D([0], [0], linewidth=1.5, linestyle='--', color='k', label='new bounds')]
    else:
        legend_elements = [Patch(facecolor='r', edgecolor='w', label='invalid area', alpha=0.2),
                           Patch(facecolor='g', edgecolor='w', label='valid area', alpha=0.2),
                           Patch(facecolor='b', edgecolor='w', label='awake area', alpha=0.2),
                           Line2D([0], [0], linewidth=1.5, linestyle='--', color='k', label='new bounds')]

    ax.legend(handles=legend_elements, loc='upper left')

    end = time.time()
    print('execution time =', (end - start), 'sec')
    print('--------')

    # plt.savefig(os.path.dirname(os.path.abspath('run_script.py')) + '/invalid_plt.png')
    plt.show()

    # return new_start, new_end, duration, valid_hours, valid_rate
