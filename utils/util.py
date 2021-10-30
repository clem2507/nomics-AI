import os
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
from tensorflow.keras.models import load_model


def convertStringToTime(start_time, start_date):
    temp_list = list(start_time)
    temp_list[len(temp_list) - 4] = '.'
    start_time = "".join(temp_list)
    date_time = start_date + ' ' + start_time
    form = '%d.%m.%y %H:%M:%S.%f'
    return dt.strptime(date_time, form)


def extractDataFromLine(line):
    out = []
    temp = line.split(';')
    # start date and time
    out.append(convertStringToTime(temp[0], temp[1]))
    # end date and time
    out.append(convertStringToTime(temp[2], temp[3]))
    # zone type
    out.append(temp[len(temp) - 1][:len(temp[len(temp) - 1]) - 1])
    return out


def timesConvertion(times, start_time, start_date):
    out = []
    start_datetime = convertStringToTime(start_time, start_date)
    for i in range(len(times)):
        temp_time = start_datetime + datetime.timedelta(0, times[i])
        out.append(temp_time)
    return out


def encoder(df):
    for i in range(len(df['label'])):
        if df.at[i, 'label'] == "Out of Range":
            df.at[i, 'label'] = 0
        else:
            df.at[i, 'label'] = 1


def checkOnlyDuplicate(lst):
    temp = lst[0]
    for x in lst:
        if temp != x:
            return False
        temp = x
    return True


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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
        temp = [curr_time, curr_time + datetime.timedelta(minutes=time_step), label]
        classes_df = classes_df.append(pd.Series(temp, index=classes_df.columns), ignore_index=True)
        curr_time += datetime.timedelta(minutes=time_step)

    flag = False
    new_start_time = None
    new_end_time = None
    while flag is False:
        if new_start_time is None or flag is False:
            for idx, row in classes_df.iterrows():
                if row.label == 0:
                    classes_df.drop(index=idx, inplace=True)
                if row.label == 1:
                    new_start_time = row.start_time
                    break

        if new_end_time is None or flag is False:
            for idx, row in classes_df.iloc[::-1].iterrows():
                if row.label == 0:
                    classes_df.drop(index=idx, inplace=True)
                if row.label == 1:
                    new_end_time = row.end_time
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

    return new_start_time, new_end_time


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


def sequence_test(model_path):
    # model loader
    model = load_model(model_path, compile=True, custom_objects={'f1_m': f1_m})

    # edf file
    file1 = r'/content/drive/MyDrive/Thesis/data/01/2019_01_31_23_56_20_121-SER-14-372(R2)_FR/2019_01_31_23_56_20_121-SER-14-372(R2)_FR'
    file2 = r'/content/drive/MyDrive/Thesis/data/01/2019_01_08_22_13_32_121-SER-15-407(R1)_FR_38y/2019_01_08_22_13_32_121-SER-15-407(R1)_FR_38y'
    file3 = r'/content/drive/MyDrive/Thesis/data/01/2019_01_30_00_55_05_121-SER-16-495(R1)_FR_69y/2019_01_30_00_55_05_121-SER-16-495(R1)_FR_69y'

    files = [file1, file2, file3]

    invalid_dir = r'/content/drive/MyDrive/Thesis/data/invalid_analysis'

    # for f in files:
    for f in os.listdir(invalid_dir):
        # make sure file name is not invalid (had issue with .DS_Store file)
        if f.startswith('20'):
            # mk3 file
            mk3_file = os.path.join(invalid_dir, f + '/' + f + '.mk3')
            data_mk3 = open(mk3_file)
            # data_mk3 = open(f+'.mk3')
            lines = data_mk3.readlines()
            start_record_time = lines[5].split(';')[0]
            start_record_date = lines[5].split(';')[1]

            # edf file
            edf_file = os.path.join(invalid_dir, f + '/' + f + '.edf')
            raw_data = mne.io.read_raw_edf(edf_file)
            # raw_data = mne.io.read_raw_edf(f+'.edf')
            data, times = raw_data[:]
            sec_interval = 0.1
            times = timesConvertion(times, start_record_time, start_record_date)
            df_jawac = pd.DataFrame()
            df_jawac.insert(0, 'times', times)
            df_jawac.insert(1, 'data', data[0])
            df_jawac = df_jawac.resample(str(sec_interval) + 'S', on='times').median()['data'].to_frame(name='data')
            df_jawac.reset_index(inplace=True)

            # time step in minutes
            time_step = 3
            count = 0
            size = int((60 / sec_interval) * time_step)
            X_test_seq = np.array([df_jawac.loc[i:i + size - 1, :].data for i in range(0, len(df_jawac), size)])

            X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, padding='post', dtype='float64')

            X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))

            df_jawac.set_index('times', inplace=True)

            predictions = model.predict(X_test_seq_pad)
            classes = []
            for item in predictions:
                idx = np.argmax(item)
                # if item[idx] > 0.9:
                classes.append(idx)
                # else:
                # classes.append(1)

            valid_total = 0
            invalid_total = 0
            for label in classes:
                if label == 1:
                    valid_total += 1
                if label == 0:
                    invalid_total += 1
            valid_mean = valid_total / len(classes)
            invalid_mean = invalid_total / len(classes)

            print('--------')

            print('valid percentage:', (valid_mean * 100), '%')
            print('invalid percentage:', (invalid_mean * 100), '%')

            print('--------')

            print('valid hours:', ((valid_total * time_step) / 60), 'h')
            print('invalid hours:', ((invalid_total * time_step) / 60), 'h')

            print('--------')

            new_start, new_end = analysis_cutting(classes, df_jawac.index[0], df_jawac.index[-1], time_step, threshold=0.8)

            print('new start analysis time:', new_start)
            print('new end analysis time:', new_end)
            if new_start is not None and new_end is not None:
                print('analysis duration:', (new_end - new_start))
            else:
                print('analysis duration: Unknown')

            print('--------')

            # graph
            fig, ax = plt.subplots()
            fig.set_size_inches(18.5, 10.5)

            ax.plot(df_jawac.resample('1S').median()['data'].to_frame(name='data').index.tolist(), df_jawac.resample('1S').median()['data'].to_frame(name='data').data.tolist())
            # ax.plot(df_jawac.index.tolist(), df_jawac.data.tolist())
            ax.axhline(y=0, color='r', linewidth=1)
            if new_start is not None and new_end is not None:
                ax.axvline(x=new_start, color='k', linewidth=2, linestyle='--')
                ax.axvline(x=new_end, color='k', linewidth=2, linestyle='--')
            ax.set(xlabel='time (s)', ylabel='opening (mm)', title='Jawac Signal')
            ax.grid()

            curr_time = convertStringToTime(start_record_time, start_record_date)
            for label in classes:
                if label == 0:
                    plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_step), facecolor='r', alpha=0.20)
                elif label == 1:
                    plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_step), facecolor='g', alpha=0.20)
                curr_time += datetime.timedelta(minutes=time_step)

            legend_elements = [Patch(facecolor='r', edgecolor='w', label='invalid area', alpha=0.2),
                               Patch(facecolor='g', edgecolor='w', label='valid area', alpha=0.2)]
            ax.legend(handles=legend_elements, loc='upper left')

            plt.show()
