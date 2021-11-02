import os
import mne
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from utils.util import extract_data_from_line, string_datetime_conversion, remove_outlier, occurrences_counter, check_nan


def load_data(time_split, time_interval):
    print(f'{time_split} minutes signal time split')
    print(f'{time_interval} minutes data resampling')
    print('-----')
    print('data loading...')
    X = np.loadtxt(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/split_{time_split}_resampling_{time_interval}/X.txt')
    y = np.loadtxt(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/split_{time_split}_resampling_{time_interval}/y.txt')

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('----')
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print('----')

    return X_train, X_test, y_train, y_test


def create_dataframes():
    # invalid analysis upload
    invalid_dir = os.path.dirname(os.path.abspath('runscript.py')) + '/data/invalid_analysis'
    invalid_count = 0
    for filename in os.listdir(invalid_dir):
        # make sure file name is not invalid (had issue with .DS_Store file)
        if not filename.startswith('.'):
            mk3_file = os.path.join(invalid_dir, filename + '/' + filename + '.mk3')
            data_mk3 = open(mk3_file)
            lines = data_mk3.readlines()
            start_record_time = lines[5].split(';')[0]
            start_record_date = lines[5].split(';')[1]
            lines = lines[7:]
            col_names = ['start', 'end', 'label']
            df_mk3 = pd.DataFrame(columns=col_names)
            for line in lines:
                if line.split(sep=';')[-1][:-1] == 'Out of Range':
                    if line.split(sep=';')[-2] == '1':
                        temp = pd.Series(extract_data_from_line(line), index=df_mk3.columns)
                        df_mk3 = df_mk3.append(temp, ignore_index=True)
            if len(df_mk3) < 1:
                continue
            else:
                df_mk3.to_pickle(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/invalid_mk3_dfs/{filename}_mk3.pkl')

            edf_file = os.path.join(invalid_dir, filename + '/' + filename + '.edf')
            raw_data = mne.io.read_raw_edf(edf_file)
            data, times = raw_data[:]
            times = string_datetime_conversion(times, start_record_time, start_record_date)
            df_jawac = pd.DataFrame()
            df_jawac.insert(0, 'times', times)
            df_jawac.insert(1, 'data', data[0])
            df_jawac.to_pickle(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/invalid_jawac_dfs/{filename}_jawac.pkl')
            invalid_count += 1
            print('invalid count =', invalid_count)

    # valid analysis upload
    valid_dir = os.path.dirname(os.path.abspath('runscript.py')) + '/data/all_analysis'
    valid_count = 0
    for filename in os.listdir(valid_dir):
        # make sure file name is not invalid (had issue with .DS_Store file)
        if not filename.startswith('.'):
            # in this case, we want to have a balanced amount of valid and invalid analysis
            if valid_count < invalid_count:
                mk3_file = os.path.join(valid_dir, filename + '/' + filename + '.mk3')
                data_mk3 = open(mk3_file)
                lines = data_mk3.readlines()
                start_record_time = lines[5].split(';')[0]
                start_record_date = lines[5].split(';')[1]
                lines = lines[7:]
                for line in lines:
                    # check if the analysis is really valid
                    # otherwise, we just skip it
                    if line.split(sep=';')[-1][:-1] == 'Out of Range':
                        continue

                edf_file = os.path.join(valid_dir, filename + '/' + filename + '.edf')
                raw_data = mne.io.read_raw_edf(edf_file)
                data, times = raw_data[:]
                times = string_datetime_conversion(times, start_record_time, start_record_date)
                df_jawac = pd.DataFrame()
                df_jawac.insert(0, 'times', times)
                df_jawac.insert(1, 'data', data[0])
                df_jawac = remove_outlier(df_jawac, 10)
                df_jawac.to_pickle(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/valid_jawac_dfs/{filename}_jawac.pkl')
                valid_count += 1
                print('valid count =', valid_count)
            else:
                break


class Preprocessing:

    def __init__(self, time_split, time_interval):
        self.invalid_jawac_df_list = []
        self.invalid_mk3_df_list = []
        self.valid_jawac_df_list = []
        self.dataset_df = pd.DataFrame(columns=['data', 'label'])
        # time_split and resampling time_interval in minutes
        self.time_split = time_split
        self.time_interval = time_interval

    def split_dataframe(self):
        count = 0
        for pkl in sorted(os.listdir(os.path.dirname(os.path.abspath('runscript.py')) + '/data/dataset/invalid_mk3_dfs')):
            if not pkl.startswith('.'):
                self.invalid_mk3_df_list.append(pd.read_pickle(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/invalid_mk3_dfs/{pkl}'))
                count += 1
                print(f'invalid mk3 #{count}')

        print('-----')
        count = 0
        for pkl in sorted(os.listdir(os.path.dirname(os.path.abspath('runscript.py')) + '/data/dataset/invalid_jawac_dfs')):
            if not pkl.startswith('.'):
                self.invalid_jawac_df_list.append(pd.read_pickle(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/invalid_jawac_dfs/{pkl}'))
                count += 1
                print(f'invalid jawac #{count}')

        print('-----')
        count = 0
        for pkl in sorted(os.listdir(os.path.dirname(os.path.abspath('runscript.py')) + '/data/dataset/valid_jawac_dfs')):
            if not pkl.startswith('.'):
                self.valid_jawac_df_list.append(pd.read_pickle(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/valid_jawac_dfs/{pkl}'))
                count += 1
                print(f'valid jawac #{count}')
        print('-----')

        col_names = ['data', 'label']
        temp_dataset_df = pd.DataFrame(columns=col_names)
        print('invalid df extraction...')
        for i in range(len(self.invalid_mk3_df_list)):
            self.invalid_jawac_df_list[i] = self.invalid_jawac_df_list[i].resample(str(self.time_interval) + 'S', on='times').median()['data'].to_frame(name='data')
            for idx, row in self.invalid_mk3_df_list[i].iterrows():
                temp = [self.invalid_jawac_df_list[i].loc[row['start']:row['end']].data.tolist(), 0]
                temp_dataset_df = temp_dataset_df.append(pd.Series(temp, index=temp_dataset_df.columns), ignore_index=True)

        print('valid df extraction...')
        for i in range(len(self.valid_jawac_df_list)):
            self.valid_jawac_df_list[i] = self.valid_jawac_df_list[i].resample(str(self.time_interval) + 'S', on='times').median()['data'].to_frame(name='data')
            temp = [self.valid_jawac_df_list[i].data.tolist(), 1]
            temp_dataset_df = temp_dataset_df.append(pd.Series(temp, index=temp_dataset_df.columns), ignore_index=True)

        print('dataset df splitting and fusion...')
        max_length = int(self.time_split * (60 / self.time_interval))
        for idx, row in temp_dataset_df.iterrows():
            temp = [row[0][i:i + max_length] for i in range(0, len(row[0]), max_length)][:-1]
            for arr in temp:
                out = [arr, row.label]
                self.dataset_df = self.dataset_df.append(pd.Series(out, index=self.dataset_df.columns), ignore_index=True)
        print('-----')

        # create directory
        os.mkdir(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/split_{self.time_split}_resampling_{self.time_interval}')

        df_info_file = open(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/split_{self.time_split}_resampling_{self.time_interval}/info.txt', 'w')
        df_info_file.write('This file contains information about the content of the X.txt file \n')
        df_info_file.write('--- \n')
        df_info_file.write(f'Splitting time in minutes = {self.time_split} \n')
        df_info_file.write(f'Resampling time in minutes = {self.time_interval} \n')
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

        self.dataset_df = self.dataset_df[check_nan(self.dataset_df['data'].values)]

        print(self.dataset_df.label.value_counts())
        print('-----')
        print('successfully saved!')

    def create_dataset(self):
        dir_path = f'split_{self.time_split}_resampling_{self.time_interval}'
        if dir_path in os.listdir(os.path.dirname(os.path.abspath('runscript.py')) + '/data/dataset'):
            X_train, y_train, X_test, y_test = load_data(time_split=self.time_split, time_interval=self.time_interval)
        else:
            self.split_dataframe()

            X = np.array(self.dataset_df.data.tolist())
            X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float64', padding='post')
            y = to_categorical(np.array(self.dataset_df['label'].tolist()))

            print('before balancing dataset:', occurrences_counter(y))

            oversample = SMOTE()
            X, y = oversample.fit_resample(X.reshape(X.shape[0], -1), y)

            y = to_categorical(y)

            print('after balancing dataset:', occurrences_counter(y))

            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            # X and y variables saving
            np.savetxt(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/split_{self.time_split}_resampling_{self.time_interval}/X.txt', X.reshape(X.shape[0], -1))
            np.savetxt(os.path.dirname(os.path.abspath('runscript.py')) + f'/data/dataset/split_{self.time_split}_resampling_{self.time_interval}/y.txt', y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            print('----')
            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)
            print('----')

        return X_train, y_train, X_test, y_test
