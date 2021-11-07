import os
import mne
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import statistics as stat

from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from utils.util import extract_data_from_line, string_datetime_conversion, remove_outlier, occurrences_counter, check_nan, block_print, enable_print


def load_data(time_split, time_resampling, num_class):
    print(f'{time_split} minutes signal time split')
    print(f'{time_resampling} minutes data resampling')
    print(f'{num_class} classes classification')
    print('-----')
    print('data loading...')

    if num_class == 2:
        X = np.loadtxt(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/binomial/split_{time_split}_resampling_{time_resampling}/X.txt')
        y = np.loadtxt(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/binomial/split_{time_split}_resampling_{time_resampling}/y.txt')
    else:
        X = np.loadtxt(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/multinomial/split_{time_split}_resampling_{time_resampling}/X.txt')
        y = np.loadtxt(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/multinomial/split_{time_split}_resampling_{time_resampling}/y.txt')

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('----')
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print('----')

    return X_train, X_test, y_train, y_test


def create_dataframes(directory):
    # analysis upload into dataframes
    directory_path = os.path.dirname(directory)
    dir_names = sorted(os.listdir(directory_path))
    for i in tqdm(range(len(dir_names))):
        # make sure file name is not invalid (had issue with the .DS_Store file)
        if not dir_names[i].startswith('.'):
            mk3_file = f'{directory_path}/{dir_names[i]}/{dir_names[i]}.mk3'
            edf_file = f'{directory_path}/{dir_names[i]}/{dir_names[i]}.edf'
            if os.path.exists(mk3_file) and os.path.exists(edf_file):
                if not os.path.exists(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/edf_dfs/{dir_names[i]}'):
                    os.mkdir(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/edf_dfs/{dir_names[i]}')
                else:
                    continue
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
                    else:
                        temp = pd.Series(extract_data_from_line(line), index=df_mk3.columns)
                        df_mk3 = df_mk3.append(temp, ignore_index=True)
                if len(df_mk3) < 1:
                    continue
                else:
                    df_mk3.to_pickle(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/edf_dfs/{dir_names[i]}/{dir_names[i]}_mk3.pkl')

                block_print()
                raw_data = mne.io.read_raw_edf(edf_file)
                enable_print()
                data, times = raw_data[:]
                times = string_datetime_conversion(times, start_record_time, start_record_date)
                df_jawac = pd.DataFrame()
                df_jawac.insert(0, 'times', times)
                df_jawac.insert(1, 'data', data[0])
                df_jawac = df_jawac.resample('0.1S', on='times').median()['data'].to_frame(name='data')
                df_jawac.to_pickle(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/edf_dfs/{dir_names[i]}/{dir_names[i]}_jawac.pkl')


class Preprocessing:

    def __init__(self, time_split, time_resampling, num_class):
        self.invalid_jawac_df_list = []
        self.invalid_mk3_df_list = []
        self.valid_jawac_df_list = []
        self.valid_mk3_df_list = []
        self.dataset_df = pd.DataFrame(columns=['data', 'label'])
        # time_split and resampling time_resampling in minutes
        self.time_split = time_split
        self.time_resampling = time_resampling
        self.num_class = num_class

    def split_dataframe(self):
        dir_names = sorted(os.listdir(os.path.dirname(os.path.abspath('classify_jawac.py')) + '/data/edf_dfs'))
        for i in tqdm(range(len(dir_names))):
            if not dir_names[i].startswith('.'):
                df_mk3 = pd.read_pickle(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/edf_dfs/{dir_names[i]}/{dir_names[i]}_mk3.pkl')
                df_jawac = pd.read_pickle(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/edf_dfs/{dir_names[i]}/{dir_names[i]}_jawac.pkl')
                if 'Out of Range' in df_mk3.label.tolist():
                    self.invalid_mk3_df_list.append(df_mk3)
                    self.invalid_jawac_df_list.append(df_jawac)
                else:
                    self.valid_mk3_df_list.append(df_mk3)
                    self.valid_jawac_df_list.append(df_jawac)

        col_names = ['data', 'label']
        temp_dataset_df = pd.DataFrame(columns=col_names)

        for i in tqdm(range(len(self.invalid_mk3_df_list))):
            self.invalid_jawac_df_list[i] = self.invalid_jawac_df_list[i].resample(str(self.time_resampling) + 'S').median()['data'].to_frame(name='data')
            for idx, row in self.invalid_mk3_df_list[i].iterrows():
                if row.label == 'Out of Range':
                    temp = [self.invalid_jawac_df_list[i].loc[row['start']:row['end']].data.tolist(), 0]
                    temp_dataset_df = temp_dataset_df.append(pd.Series(temp, index=temp_dataset_df.columns), ignore_index=True)

        for i in tqdm(range(len(self.valid_mk3_df_list))):
            self.valid_jawac_df_list[i] = self.valid_jawac_df_list[i].resample(str(self.time_resampling) + 'S').median()['data'].to_frame(name='data')
            if self.num_class == 2:
                temp = [self.valid_jawac_df_list[i].data.tolist(), 1]
                temp_dataset_df = temp_dataset_df.append(pd.Series(temp, index=temp_dataset_df.columns), ignore_index=True)
            else:
                for idx, row in self.valid_mk3_df_list[i].iterrows():
                    if row.label == "Zone d'exclusion":
                        temp = [self.valid_jawac_df_list[i].loc[row['start']:row['end']].data.tolist(), 2]
                        self.valid_jawac_df_list[i].drop(self.valid_jawac_df_list[i].loc[row['start']:row['end']].index.tolist(), inplace=True)
                        temp_dataset_df = temp_dataset_df.append(pd.Series(temp, index=temp_dataset_df.columns), ignore_index=True)

                prev_time = None
                temp = []
                for idx, row in self.valid_jawac_df_list[i].iterrows():
                    if prev_time is not None:
                        if prev_time + datetime.timedelta(0, seconds=self.time_resampling) == idx:
                            temp.append(row.data)
                        else:
                            if len(temp) > 0:
                                temp_dataset_df = temp_dataset_df.append(pd.Series([temp, 1], index=temp_dataset_df.columns), ignore_index=True)
                            temp = []
                    prev_time = idx
                if len(temp) > 0:
                    temp_dataset_df = temp_dataset_df.append(pd.Series([temp, 1], index=temp_dataset_df.columns), ignore_index=True)

        max_length = int(self.time_split * (60 / self.time_resampling))
        print(f'dataset df splitting in samples of length {max_length}...')
        div, step, step_num = 10, 0, 0
        step_max = int(len(temp_dataset_df) / div)
        curr_time = time.time()
        time_list = []
        for idx, row in temp_dataset_df.iterrows():
            temp = [row[0][i:i + max_length] for i in range(0, len(row[0]), max_length)][:-1]
            for arr in temp:
                out = [arr, row.label]
                self.dataset_df = self.dataset_df.append(pd.Series(out, index=self.dataset_df.columns), ignore_index=True)
            step += 1
            if step > step_max:
                step = 0
                step_num += 1
                time_list.append(time.time() - curr_time)
                print(f'progress --> {step_num}/{div} | time: {round(time.time() - curr_time, 2)} sec | est. rem. time: {round((stat.mean(time_list)) * (div - step_num), 2)} sec / {round(((stat.mean(time_list)) * (div - step_num)) / 60, 2)} min |')
                curr_time = time.time()
            # print(f'{round(time.time() - curr_time, 2)} sec', end='\r')
        time_list.append(time.time() - curr_time)
        print(f'progress --> {div}/{div} | total time: {round(sum(time_list) / 60, 2)} min')

        print('-----')

        # create directory
        if self.num_class == 2:
            os.mkdir(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/binomial/split_{self.time_split}_resampling_{self.time_resampling}')
            df_info_file = open(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/binomial/split_{self.time_split}_resampling_{self.time_resampling}/info.txt', 'w')
        else:
            os.mkdir(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/multinomial/split_{self.time_split}_resampling_{self.time_resampling}')
            df_info_file = open(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/multinomial/split_{self.time_split}_resampling_{self.time_resampling}/info.txt', 'w')

        df_info_file.write('This file contains information about the content of the X.txt file \n')
        df_info_file.write('--- \n')
        df_info_file.write(f'Splitting time in minutes = {self.time_split} \n')
        df_info_file.write(f'Resampling time in seconds = {self.time_resampling} \n')
        df_info_file.write(f'Sample size = {max_length} \n')
        df_info_file.write(f'Num of samples in total = {len(self.dataset_df)} \n')
        if 0 in self.dataset_df.label.value_counts().keys():
            df_info_file.write(f'Num of invalid samples = {self.dataset_df.label.value_counts()[0]} \n')
        else:
            df_info_file.write('Num of invalid samples = 0 \n')
        if 1 in self.dataset_df.label.value_counts().keys():
            df_info_file.write(f'Num of valid samples = {self.dataset_df.label.value_counts()[1]} \n')
        else:
            df_info_file.write('Num of valid samples = 0 \n')
        if self.num_class != 2:
            if 2 in self.dataset_df.label.value_counts().keys():
                df_info_file.write(f'Num of awake samples = {self.dataset_df.label.value_counts()[2]} \n')
            else:
                df_info_file.write('Num of awake samples = 0 \n')
        df_info_file.write('---')
        df_info_file.close()

        self.dataset_df = self.dataset_df[check_nan(self.dataset_df['data'].values)]

        print(self.dataset_df.label.value_counts())
        print('-----')

    def create_dataset(self):
        split_flag = False
        dir_path = f'split_{self.time_split}_resampling_{self.time_resampling}'
        if self.num_class == 2:
            if dir_path in os.listdir(os.path.dirname(os.path.abspath('classify_jawac.py')) + '/data/samples/binomial'):
                X_train, y_train, X_test, y_test = load_data(time_split=self.time_split, time_resampling=self.time_resampling, num_class=self.num_class)
                # TODO uncomment after experiments
                # self.split_dataframe()
                # split_flag = True
            else:
                self.split_dataframe()
                split_flag = True
        else:
            if dir_path in os.listdir(os.path.dirname(os.path.abspath('classify_jawac.py')) + '/data/samples/multinomial'):
                X_train, y_train, X_test, y_test = load_data(time_split=self.time_split, time_resampling=self.time_resampling, num_class=self.num_class)
                # TODO uncomment after experiments
                # self.split_dataframe()
                # split_flag = True
            else:
                self.split_dataframe()
                split_flag = True

        if split_flag:
            X = np.array(self.dataset_df.data.tolist())
            X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float64', padding='post')
            y = to_categorical(np.array(self.dataset_df['label'].tolist()))

            print('before balancing dataset:', occurrences_counter(y))

            oversample = SMOTE()
            X, y = oversample.fit_resample(X.reshape(X.shape[0], -1), y)

            if self.num_class == 2:
                y = to_categorical(y)

            print('after balancing dataset:', occurrences_counter(y))

            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            # X and y variables saving
            if self.num_class == 2:
                np.savetxt(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/binomial/split_{self.time_split}_resampling_{self.time_resampling}/X.txt', X.reshape(X.shape[0], -1))
                np.savetxt(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/binomial/split_{self.time_split}_resampling_{self.time_resampling}/y.txt', y)
            else:
                np.savetxt(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/multinomial/split_{self.time_split}_resampling_{self.time_resampling}/X.txt', X.reshape(X.shape[0], -1))
                np.savetxt(os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/data/samples/multinomial/split_{self.time_split}_resampling_{self.time_resampling}/y.txt', y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            print('----')
            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)
            print('----')

        return X_train, y_train, X_test, y_test
