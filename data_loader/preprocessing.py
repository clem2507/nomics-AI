import os
import mne
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from utils.util import extractDataFromLine, timesConvertion, remove_outlier, occurrences_counter


def load_data():
    X = np.loadtxt('data/dataset/X.txt')
    y = np.loadtxt('data/dataset/y.txt')

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('----')
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print('----')

    return X_train, X_test, y_train, y_test


class Preprocessing:

    def __init__(self):
        self.invalid_jawac_df_list = []
        self.invalid_mk3_df_list = []
        self.valid_jawac_df_list = []
        self.dataset_df = pd.DataFrame(columns=['data', 'label'])
        # do not change this value
        self.sec_interval = 0.1

    def load_dataframe(self):
        self.dataset_df = pd.read_pickle('data/dataset/df.pkl')

    def create_dataframe(self):
        # do not change this value
        # invalid analysis upload
        invalid_dir = 'data/invalid_analysis'
        for filename in os.listdir(invalid_dir):
            # make sure file name is not invalid (had issue with .DS_Store file)
            if filename.startswith('20'):
                mk3_file = os.path.join(invalid_dir, filename + '/' + filename + '.mk3')
                data_mk3 = open(mk3_file)
                lines = data_mk3.readlines()
                start_record_time = lines[5].split(';')[0]
                start_record_date = lines[5].split(';')[1]
                lines = lines[7:]
                col_names = ['start', 'end', 'label']
                df_mk3 = pd.DataFrame(columns=col_names)
                for line in lines:
                    if line.split(sep=";")[-1][:-1] == 'Out of Range':
                        if line.split(sep=";")[-2] == '1':
                            temp = pd.Series(extractDataFromLine(line), index=df_mk3.columns)
                            df_mk3 = df_mk3.append(temp, ignore_index=True)
                if len(df_mk3) < 1:
                    continue
                else:
                    self.invalid_mk3_df_list.append(df_mk3)

                edf_file = os.path.join(invalid_dir, filename + '/' + filename + '.edf')
                raw_data = mne.io.read_raw_edf(edf_file)
                data, times = raw_data[:]
                times = timesConvertion(times, start_record_time, start_record_date)
                df_jawac = pd.DataFrame()
                df_jawac.insert(0, 'times', times)
                df_jawac.insert(1, 'data', data[0])
                # df_jawac = df_jawac.set_index('times')
                df_jawac = df_jawac.resample(str(self.sec_interval) + 'S', on='times').median()['data'].to_frame(name='data')
                # here, we do not remove outliers since we want to have them in invalid data training set
                # df_jawac = remove_outlier(df_jawac, 10)
                self.invalid_jawac_df_list.append(df_jawac)
                print('invalid jawac list size =', len(self.invalid_jawac_df_list))
        print('-------')

        # valid analysis upload
        valid_dir = 'data/all_analysis'
        for filename in os.listdir(valid_dir):
            # make sure file name is not invalid (had issue with .DS_Store file)
            if filename.startswith('20'):
                # in this case, we want to have a balanced amount of valid and invalid analysis
                if len(self.valid_jawac_df_list) < len(self.invalid_jawac_df_list):
                    mk3_file = os.path.join(valid_dir, filename + '/' + filename + '.mk3')
                    data_mk3 = open(mk3_file)
                    lines = data_mk3.readlines()
                    start_record_time = lines[5].split(';')[0]
                    start_record_date = lines[5].split(';')[1]
                    lines = lines[7:]
                    for line in lines:
                        # check if the analysis is really valid
                        # otherwise, we just skip it
                        if line.split(sep=";")[-1][:-1] == 'Out of Range':
                            continue

                    edf_file = os.path.join(valid_dir, filename + '/' + filename + '.edf')
                    raw_data = mne.io.read_raw_edf(edf_file)
                    data, times = raw_data[:]
                    times = timesConvertion(times, start_record_time, start_record_date)
                    df_jawac = pd.DataFrame()
                    df_jawac.insert(0, 'times', times)
                    df_jawac.insert(1, 'data', data[0])
                    # df_jawac = df_jawac.set_index('times')
                    df_jawac = df_jawac.resample(str(self.sec_interval) + 'S', on='times').median()['data'].to_frame(name='data')
                    df_jawac = remove_outlier(df_jawac, 10)
                    self.valid_jawac_df_list.append(df_jawac)
                    print('valid jawac list size =', len(self.valid_jawac_df_list))
                else:
                    break

    def split_dataframe(self):
        col_names = ['data', 'label']
        temp_dataset_df = pd.DataFrame(columns=col_names)
        for i in range(len(self.invalid_mk3_df_list)):
            for idx, row in self.invalid_mk3_df_list[i].iterrows():
                temp = [self.invalid_jawac_df_list[i].loc[row['start']:row['end']].data.tolist(), 0]
                temp_dataset_df = temp_dataset_df.append(pd.Series(temp, index=temp_dataset_df.columns), ignore_index=True)

        for i in range(len(self.valid_jawac_df_list)):
            temp = [self.valid_jawac_df_list[i].data.tolist(), 1]
            temp_dataset_df = temp_dataset_df.append(pd.Series(temp, index=temp_dataset_df.columns), ignore_index=True)

        # time split in minutes
        time_split = 3
        max_length = time_split * 600
        for idx, row in temp_dataset_df.iterrows():
            temp = [row[0][i:i + max_length] for i in range(0, len(row[0]), max_length)][:-1]
            for arr in temp:
                out = [arr, row.label]
                self.dataset_df = self.dataset_df.append(pd.Series(out, index=self.dataset_df.columns), ignore_index=True)

        self.dataset_df.to_pickle('data/dataset/df.pkl')

        df_info_file = open('data/dataset/df_info.txt', 'w')
        df_info_file.write('This file contains information about the currently saved dataset df \n')
        df_info_file.write('--- \n')
        df_info_file.write(f'Sampling time in minutes = {time_split} \n')
        df_info_file.write(f'Sample size = {max_length} \n')
        df_info_file.write(f'Num of samples in total = {len(self.dataset_df)} \n')
        if 0 in self.dataset_df.label.value_counts().keys():
            df_info_file.write(f'Num of invalid samples = {self.dataset_df.label.value_counts()[0]} \n')
        else:
            df_info_file.write(f'Num of invalid samples = 0 \n')
        if 1 in self.dataset_df.label.value_counts().keys():
            df_info_file.write(f'Num of valid samples = {self.dataset_df.label.value_counts()[1]} \n')
        else:
            df_info_file.write(f'Num of valid samples = 0 \n')
        df_info_file.write('---')
        df_info_file.close()
        print('---')
        print(self.dataset_df.label.value_counts())
        print('---')

    def create_dataset(self, create_flag):
        if create_flag:
            self.create_dataframe()
            self.split_dataframe()
        else:
            self.load_dataframe()

        X = np.array(self.dataset_df.data.tolist())
        X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float64', padding='post')
        y = to_categorical(np.array(self.dataset_df['label'].tolist()))

        print('before balancing dataset:', occurrences_counter(y))

        oversample = SMOTE()
        X, y = oversample.fit_resample(X.reshape(X.shape[0], -1), y)

        y = to_categorical(y)

        print('after balancing dataset:', occurrences_counter(y))

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # X and y variables saving in google drive
        np.savetxt('data/dataset/X.txt', X.reshape(X.shape[0], -1))
        np.savetxt('data/dataset/y.txt', y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        print('----')
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        print('----')

        return X_train, y_train, X_test, y_test
