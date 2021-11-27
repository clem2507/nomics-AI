import os
import mne
import sys
import time
import shutil
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/utils')

from util import extract_data_from_line, string_datetime_conversion, occurrences_counter, check_nan, block_print, enable_print


def load_data(segmentation_value, downsampling_value, num_class, data_balancing):
    """
    Method used to load the data already processed and saved for model's training

    Parameters:

    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -num_class: number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -data_balancing: true if balanced data is needed, false otherwise

    Returns:

    -X_train: x data instances for training the model
    -y_train: y label instances for training the model
    -X_test: x data instances for testing the model
    -y_test: y label instances for testing the model
    """

    print('load data...')

    if data_balancing:
        is_balanced = 'balanced'
    else:
        is_balanced = 'unbalanced'

    # data loader
    if num_class == 2:
        X = np.loadtxt(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{is_balanced}/binary/split_{segmentation_value}_resampling_{downsampling_value}/X.txt')
        y = np.loadtxt(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{is_balanced}/binary/split_{segmentation_value}_resampling_{downsampling_value}/y.txt')
    else:
        X = np.loadtxt(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{is_balanced}/multinomial/split_{segmentation_value}_resampling_{downsampling_value}/X.txt')
        y = np.loadtxt(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{is_balanced}/multinomial/split_{segmentation_value}_resampling_{downsampling_value}/y.txt')

    # reshaping the data to suit the training process
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # splitting the data into testing and training sets
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('----')
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print('----')

    return X_train, X_test, y_train, y_test


def create_dataframes(directory):
    """
    Function that update, creates and saves the dataframes based on the analyses files

    Parameters:

    -directory: directory path containing the analyses to create the dataframes
    """

    dir_names = sorted(os.listdir(directory))    # analyses directory names
    for i in tqdm(range(len(dir_names))):
        # make sure file name is not invalid (had issue with the .DS_Store file)
        if not dir_names[i].startswith('.'):
            mk3_file = f'{directory}/{dir_names[i]}/{dir_names[i]}.mk3'    # mk3 file path containing inforamtion about the analysis label
            edf_file = f'{directory}/{dir_names[i]}/{dir_names[i]}.edf'    # edf file path containing time series data
            if os.path.exists(mk3_file) and os.path.exists(edf_file):
                if not os.path.exists(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/edf_dfs/{dir_names[i]}'):
                    # create the analysis directory to save the dataframes if it does not already exist
                    os.mkdir(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/edf_dfs/{dir_names[i]}')
                else:
                    continue
                data_mk3 = open(mk3_file)    # mk3 file reading
                lines = data_mk3.readlines()    # list with the mk3 lines
                start_record_time = lines[5].split(';')[0]
                start_record_date = lines[5].split(';')[1]
                lines = lines[7:]    # log file information removed (not needed anymore)
                col_names = ['start', 'end', 'label']
                df_mk3 = pd.DataFrame(columns=col_names)    # dataframe with label data created
                for line in lines:
                    # Out of Range is the label for invalid signal area
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
                    # dataframe saving
                    df_mk3.to_pickle(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/edf_dfs/{dir_names[i]}/{dir_names[i]}_mk3.pkl')

                block_print()
                raw_data = mne.io.read_raw_edf(edf_file)    # edf file reading
                enable_print()

                data, times = raw_data[:]    # edf file data extraction
                times = string_datetime_conversion(times, start_record_time, start_record_date)    # conversion to usable date dtype 
                df_jawac = pd.DataFrame()    # dataframe creation to store time series data
                df_jawac.insert(0, 'times', times)
                df_jawac.insert(1, 'data', data[0])
                # downsamping signal to 10Hz to have normalized data (some devices record with higher frequency)
                df_jawac = df_jawac.resample('0.1S', on='times').median()['data'].to_frame(name='data')
                # dataframe saving
                df_jawac.to_pickle(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/edf_dfs/{dir_names[i]}/{dir_names[i]}_jawac.pkl')
    
    dfs_directory = os.path.dirname(os.path.abspath('util.py')) + '/training/data/edf_dfs'
    dfs_names = sorted(os.listdir(dfs_directory))
    # removal of unwanted directories in the path of the input directory
    for i in tqdm(range(len(dfs_names))):
        # make sure file name is not invalid (had issue with the .DS_Store file)
        if not dfs_names[i].startswith('.'):
            if dfs_names[i] not in dir_names:
                shutil.rmtree(dfs_directory + f'/{dfs_names[i]}')


class Preprocessing:
    """
    Class used to preprocess the created dataframes

    Attributes:

    -invalid_jawac_df_list: list of dataframes containing data about the invalid time series
    -invalid_mk3_df_list: list of dataframes containing data about the invalid labeling
    -valid_jawac_df_list: list of dataframes containing data about the valid time series
    -valid_mk3_df_list: list of dataframes containing data about the valid labeling
    -dataset_df: final dataframe containing the merged processed data
    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -num_class: number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -data_balancing: true is balanced data is needed, false otherwise
    """

    def __init__(self, segmentation_value, downsampling_value, num_class, data_balancing):
        self.invalid_jawac_df_list = []
        self.invalid_mk3_df_list = []
        self.valid_jawac_df_list = []
        self.valid_mk3_df_list = []
        self.dataset_df = pd.DataFrame(columns=['data', 'label'])
        self.segmentation_value = float(segmentation_value)
        self.downsampling_value = float(downsampling_value)
        self.num_class = num_class
        self.data_balancing = data_balancing
        if self.data_balancing:
            self.is_balanced = 'balanced'
        else:
            self.is_balanced = 'unbalanced'

    def split_dataframe(self):
        """
        Method used to segment the dataframes into one common dataframe
        """

        dir_names = sorted(os.listdir(os.path.dirname(os.path.abspath('util.py')) + '/training/data/edf_dfs'))
        # loop through the dataframe directory to differentiate analyses with invalid regions from those with only valid regions
        for i in tqdm(range(len(dir_names))):
            if not dir_names[i].startswith('.'):
                df_mk3 = pd.read_pickle(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/edf_dfs/{dir_names[i]}/{dir_names[i]}_mk3.pkl')
                df_jawac = pd.read_pickle(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/edf_dfs/{dir_names[i]}/{dir_names[i]}_jawac.pkl')
                if 'Out of Range' in df_mk3.label.tolist():
                    self.invalid_mk3_df_list.append(df_mk3)
                    self.invalid_jawac_df_list.append(df_jawac)
                else:
                    self.valid_mk3_df_list.append(df_mk3)
                    self.valid_jawac_df_list.append(df_jawac)

        col_names = ['data', 'label']    # dataframe column names
        temp_dataset_df = pd.DataFrame(columns=col_names)    # temporary dataframe creation

        # split of the invalid dataframes
        for i in tqdm(range(len(self.invalid_mk3_df_list))):
            self.invalid_jawac_df_list[i] = self.invalid_jawac_df_list[i].resample(str(self.downsampling_value) + 'S').median()['data'].to_frame(name='data')
            for idx, row in self.invalid_mk3_df_list[i].iterrows():
                if row.label == 'Out of Range':
                    temp = [self.invalid_jawac_df_list[i].loc[row['start']:row['end']].data.tolist(), 0]
                    temp_dataset_df = temp_dataset_df.append(pd.Series(temp, index=temp_dataset_df.columns), ignore_index=True)

        # split of the valid dataframes
        for i in tqdm(range(len(self.valid_mk3_df_list))):
            self.valid_jawac_df_list[i] = self.valid_jawac_df_list[i].resample(str(self.downsampling_value) + 'S').median()['data'].to_frame(name='data')
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
                        if prev_time + datetime.timedelta(0, seconds=self.downsampling_value) == idx:
                            temp.append(row.data)
                        else:
                            if len(temp) > 0:
                                temp_dataset_df = temp_dataset_df.append(pd.Series([temp, 1], index=temp_dataset_df.columns), ignore_index=True)
                            temp = []
                    prev_time = idx
                if len(temp) > 0:
                    temp_dataset_df = temp_dataset_df.append(pd.Series([temp, 1], index=temp_dataset_df.columns), ignore_index=True)

        # valid and invalid jawac dataframes merging into dataset df
        max_length = int(self.segmentation_value * (60 / self.downsampling_value))
        for idx in tqdm(range(len(temp_dataset_df))):
            temp = [temp_dataset_df.iloc[idx][0][i:i + max_length] for i in range(0, len(temp_dataset_df.iloc[idx][0]), max_length)][:-1]
            for arr in temp:
                out = [arr, temp_dataset_df.iloc[idx].label]
                self.dataset_df = self.dataset_df.append(pd.Series(out, index=self.dataset_df.columns), ignore_index=True)

        print('-----')

        if self.num_class == 2:
            # create directory
            if not os.path.exists(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/binary/split_{self.segmentation_value}_resampling_{self.downsampling_value}'):
                os.mkdir(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/binary/split_{self.segmentation_value}_resampling_{self.downsampling_value}')
            # create information file that contains information about the dataframe
            df_info_file = open(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/binary/split_{self.segmentation_value}_resampling_{self.downsampling_value}/info.txt', 'w')
        else:
            # create directory
            if not os.path.exists(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/multinomial/split_{self.segmentation_value}_resampling_{self.downsampling_value}'):
                os.mkdir(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/multinomial/split_{self.segmentation_value}_resampling_{self.downsampling_value}')
            # create information file that contains information about the dataframe
            df_info_file = open(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/multinomial/split_{self.segmentation_value}_resampling_{self.downsampling_value}/info.txt', 'w')

        df_info_file.write('This file contains information about the content of the directory \n')
        df_info_file.write('--- \n')
        df_info_file.write(f'Splitting time in minutes = {self.segmentation_value} \n')
        df_info_file.write(f'Resampling time in seconds = {self.downsampling_value} \n')
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
        df_info_file.write('--- \n')
        df_info_file.close()

        self.dataset_df = self.dataset_df[check_nan(self.dataset_df['data'].values)]

        print(self.dataset_df.label.value_counts())
        print('-----')

    def create_dataset(self):
        """
        Method that transforms the dataframe dataset into numpy array to better fit the learning models

        Returns:

        -X_train: x data instances for training the model
        -y_train: y label instances for training the model
        -X_test: x data instances for testing the model
        -y_test: y label instances for testing the model
        """

        start_time = time.time()    # start timer variable used for the calculation of the total execution time 
        print('-----')
        print(f'{self.segmentation_value} minutes signal time split')
        print(f'{self.downsampling_value} minutes data resampling')
        print(f'{self.num_class} classes classification')
        print(f'data balancing: {self.data_balancing}')
        print('-----')
        self.split_dataframe()

        X = np.array(self.dataset_df.data.tolist())    # X data instances
        # padding function to make sure everything is the same size
        X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float64', padding='post')
        # converting label list into categorical values
        y = to_categorical(np.array(self.dataset_df['label'].tolist()))

        occ_counter = occurrences_counter(y)    # classes occurences counter in the y list

        if self.num_class == 2:
            x_values = ['Invalid', 'Valid']
            y_values = [occ_counter[0], occ_counter[1]]
            dir_path = os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/binary/split_{self.segmentation_value}_resampling_{self.downsampling_value}/occ_bar_plt.png'
        else:
            x_values = ['Invalid', 'Valid', 'Awake']
            y_values = [occ_counter[0], occ_counter[1], occ_counter[2]]
            dir_path = os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/multinomial/split_{self.segmentation_value}_resampling_{self.downsampling_value}/occ_bar_plt.png'

        # bar plot with classes occurrences
        plt.bar(x_values, y_values)
        plt.title('Classes occurrences before balancing')
        plt.xlabel('classes')
        plt.ylabel('count')
        plt.savefig(dir_path)
        plt.close()

        print('before balancing dataset:', occurrences_counter(y))

        # SMOTE data balancing if desired
        if self.data_balancing:
            oversample = SMOTE()
            X, y = oversample.fit_resample(X.reshape(X.shape[0], -1), y)

            if self.num_class == 2:
                y = to_categorical(y)

        print('after balancing dataset:', occurrences_counter(y))

        # X numpy array resampling to fit model training
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # X and y variables saving
        if self.num_class == 2:
            np.savetxt(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/binary/split_{self.segmentation_value}_resampling_{self.downsampling_value}/X.txt', X.reshape(X.shape[0], -1))
            np.savetxt(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/binary/split_{self.segmentation_value}_resampling_{self.downsampling_value}/y.txt', y)
        else:
            np.savetxt(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/multinomial/split_{self.segmentation_value}_resampling_{self.downsampling_value}/X.txt', X.reshape(X.shape[0], -1))
            np.savetxt(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/multinomial/split_{self.segmentation_value}_resampling_{self.downsampling_value}/y.txt', y)

        # splitting the data into testing and training sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        end_time = time.time()    # end timer variable used for the calculation of the total execution time 
        if self.num_class == 2:
            df_info_file = open(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/binary/split_{self.segmentation_value}_resampling_{self.downsampling_value}/info.txt', 'a')
        else:
            df_info_file = open(os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.is_balanced}/multinomial/split_{self.segmentation_value}_resampling_{self.downsampling_value}/info.txt', 'a')
        df_info_file.write(f'Total data preprocessing computation time = {round(end_time-start_time, 2)} sec | {round((end_time-start_time)/60, 2)} min \n')
        df_info_file.write('---')
        df_info_file.close()

        print('----')
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        print('----')
        print(f'Total data preprocessing computation time = {round(end_time-start_time, 2)} sec | {round((end_time-start_time)/60, 2)} min')
        print('----')

        return X_train, y_train, X_test, y_test
