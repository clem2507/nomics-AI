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


class Preprocessing:
    """
    Class used to preprocess the created dataframes

    Attributes:

    -invalid_jawac_df_list: list of dataframes containing data about the invalid time series
    -invalid_mk3_df_list: list of dataframes containing data about the invalid labeling
    -valid_jawac_df_list: list of dataframes containing data about the valid time series
    -valid_mk3_df_list: list of dataframes containing data about the valid labeling
    -analysis_directory: directory path containing the analyses to create the dataframes
    -dataset_df: final dataframe containing the merged processed data
    -segmentation_value: window segmentation value in minute
    -downsampling_value: signal downsampling value in second
    -num_class: number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -data_balancing: true is balanced data is needed, false otherwise
    """

    def __init__(self, analysis_directory='', segmentation_value=1, downsampling_value=1, num_class=2, data_balancing=True, log_time=''):
        self.invalid_jawac_df_list = []
        self.invalid_mk3_df_list = []
        self.valid_jawac_df_list = []
        self.valid_mk3_df_list = []
        self.dataset_df = pd.DataFrame(columns=['data', 'label'])
        self.directory = analysis_directory
        self.dfs_directory = f'{analysis_directory}_edf_dfs'
        self.segmentation_value = float(segmentation_value)
        self.downsampling_value = float(downsampling_value)
        self.num_class = num_class
        self.data_balancing = data_balancing
        self.log_time = log_time
        self.jawac_df_list = []
        self.mk3_df_list = []
        self.dataset = []
        if self.data_balancing:
            self.is_balanced = 'balanced'
        else:
            self.is_balanced = 'unbalanced'

    
    def create_dataframes(self):
        """
        Function that update, creates and saves the dataframes based on the analyses files
        """

        dir_names = sorted(os.listdir(self.directory))    # analyses directory names
        if not os.path.exists(self.dfs_directory):
            os.mkdir(self.dfs_directory)
        print('----')
        for i in tqdm(range(len(dir_names))):
            # make sure file name is not invalid (had issue with the .DS_Store file)
            if not dir_names[i].startswith('.'):
                mk3_file = f'{self.directory}/{dir_names[i]}/{dir_names[i]}.mk3'    # mk3 file path containing inforamtion about the analysis label
                edf_file = f'{self.directory}/{dir_names[i]}/{dir_names[i]}.edf'    # edf file path containing time series data
                if os.path.exists(mk3_file) and os.path.exists(edf_file):
                    if not os.path.exists(f'{self.dfs_directory}/{dir_names[i]}'):
                        # create the analysis directory to save the dataframes if it does not already exist
                        os.mkdir(f'{self.dfs_directory}/{dir_names[i]}')
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

                    # dataframe saving
                    df_mk3.to_pickle(f'{self.dfs_directory}/{dir_names[i]}/{dir_names[i]}_mk3.pkl')

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
                    df_jawac.to_pickle(f'{self.dfs_directory}/{dir_names[i]}/{dir_names[i]}_jawac.pkl')
                else:
                    print(f'edf --> {edf_file} or mk3 file --> {mk3_file} not found')
        
        dfs_names = sorted(os.listdir(self.dfs_directory))
        # removal of unwanted directories in the path of the input directory
        for i in tqdm(range(len(dfs_names))):
            # make sure file name is not invalid (had issue with the .DS_Store file)
            if not dfs_names[i].startswith('.'):
                if dfs_names[i] not in dir_names:
                    shutil.rmtree(self.dfs_directory + f'/{dfs_names[i]}')

    def split_dataframe_cnn(self):
        """
        Method used to segment the dataframes into one common dataframe for the CNN
        """

        dir_names = sorted(os.listdir(self.dfs_directory))
        # loop through the dataframe directory to differentiate analyses with invalid regions from those with only valid regions
        for i in tqdm(range(len(dir_names))):
            if not dir_names[i].startswith('.'):
                df_mk3 = pd.read_pickle(f'{self.dfs_directory}/{dir_names[i]}/{dir_names[i]}_mk3.pkl')
                df_jawac = pd.read_pickle(f'{self.dfs_directory}/{dir_names[i]}/{dir_names[i]}_jawac.pkl')
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
                    self.invalid_jawac_df_list[i].drop(self.invalid_jawac_df_list[i].loc[row['start']:row['end']].index.tolist(), inplace=True)
                    temp_dataset_df = temp_dataset_df.append(pd.Series(temp, index=temp_dataset_df.columns), ignore_index=True)

            prev_time = None
            temp = []
            for idx, row in self.invalid_jawac_df_list[i].iterrows():
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
                arr.append(np.var(arr)*1000)
                out = [arr, temp_dataset_df.iloc[idx].label]
                self.dataset_df = self.dataset_df.append(pd.Series(out, index=self.dataset_df.columns), ignore_index=True)

        self.dataset_df = self.dataset_df[check_nan(self.dataset_df['data'].values)]

        print('-----')
        print(self.dataset_df.label.value_counts())
        print('-----')

    def create_dataset_cnn(self):
        """
        Method that transforms the dataframe dataset into numpy array to better fit the CNN learning models

        Returns:

        -X_train: x data instances for training the model
        -y_train: y label instances for training the model
        -X_test: x data instances for testing the model
        -y_test: y label instances for testing the model
        """

        start_time = time.time()    # start timer variable used for the calculation of the total execution time 
        print('-----')
        print(f'{self.segmentation_value} min signal time split')
        print(f'{self.downsampling_value} sec data resampling')
        print(f'{self.num_class} classes classification')
        print(f'data balancing: {self.data_balancing}')
        print('-----')
        self.split_dataframe_cnn()

        X = np.array(self.dataset_df.data.tolist())    # X data instances
        # padding function to make sure everything is the same size
        X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype='float64', padding='post')
        # converting label list into categorical values
        y = to_categorical(np.array(self.dataset_df['label'].tolist()))

        occ_counter = occurrences_counter(y)    # classes occurences counter in the y list

        save_dir = os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.log_time}'
        os.mkdir(save_dir)

        if self.num_class == 2:
            x_values = ['Invalid', 'Valid']
            y_values = [occ_counter[0], occ_counter[1]]
        else:
            x_values = ['Invalid', 'Valid', 'Awake']
            y_values = [occ_counter[0], occ_counter[1], occ_counter[2]]

        # bar plot with classes occurrences
        plt.bar(x_values, y_values)
        plt.title('Classes occurrences before balancing')
        plt.xlabel('classes')
        plt.ylabel('count')
        plt.savefig(f'{save_dir}/occ_bar_plt.png')
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
        # np.savetxt(f'{save_dir}/X.txt', X.reshape(X.shape[0], -1))
        # np.savetxt(f'{save_dir}/y.txt', y)

        # splitting the data into testing and training sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        end_time = time.time()    # end timer variable used for the calculation of the total execution time 

        df_info_file = open(f'{save_dir}/info.txt', 'a')
        df_info_file.write(f'Is balanced = {self.is_balanced} \n')
        df_info_file.write(f'Segmentation value = {self.segmentation_value} \n')
        df_info_file.write(f'Downsampling value = {self.downsampling_value} \n')
        df_info_file.write(f'Signal resolution = {1/self.downsampling_value} \n')
        df_info_file.write('--- \n')
        df_info_file.write(f'X train shape = {X_train.shape} \n')
        df_info_file.write(f'X test shape = {X_test.shape} \n')
        df_info_file.write(f'y train shape = {y_train.shape} \n')
        df_info_file.write(f'y test shape = {y_test.shape} \n')
        df_info_file.write('--- \n')
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

    def split_dataframe_lstm(self):
        """
        Method used to segment the dataframes into one common dataframe for the LSTM
        """

        dir_names = sorted(os.listdir(self.dfs_directory))
        for i in tqdm(range(len(dir_names))):
            if not dir_names[i].startswith('.'):
                df_mk3 = pd.read_pickle(f'{self.dfs_directory}/{dir_names[i]}/{dir_names[i]}_mk3.pkl')
                df_jawac = pd.read_pickle(f'{self.dfs_directory}/{dir_names[i]}/{dir_names[i]}_jawac.pkl')
                df_jawac = df_jawac.resample(str(self.downsampling_value) + 'S').median()['data'].to_frame(name='data')
                self.mk3_df_list.append(df_mk3)
                self.jawac_df_list.append(df_jawac)

        for i in tqdm(range(len(self.jawac_df_list))):
            self.jawac_df_list[i]['label'] = [1 for i in range(len(self.jawac_df_list[i]))]
            for idx, row in self.mk3_df_list[i].iterrows():
                mask = (self.jawac_df_list[i].index >= row.start) & (self.jawac_df_list[i].index <= row.end)
                self.jawac_df_list[i].loc[mask, ['label']] = 0
            
        # split of the dataframes
        for i in tqdm(range(len(self.jawac_df_list))):
            max_length = int(self.segmentation_value * (60 / self.downsampling_value))
            split = [self.jawac_df_list[i][idx:idx + max_length] for idx in range(0, len(self.jawac_df_list[i]), max_length)][:-1]
            temp = []
            for arr in split:
                # temp.append([arr.data.tolist(), max(arr.label.tolist(), key=arr.label.tolist().count)])
                temp.append([arr.data.tolist(), arr.label.tolist()])
                # arr.append(np.var(arr)*1000)
            self.dataset.append(temp)
            
    def create_dataset_lstm(self):
        """
        Method that transforms the dataframe dataset into numpy array to better fit the LSTM learning model

        Returns:

        -X_train: x data instances for training the model
        -y_train: y label instances for training the model
        -X_test: x data instances for testing the model
        -y_test: y label instances for testing the model
        """

        start_time = time.time()    # start timer variable used for the calculation of the total execution time 
        print('-----')
        print(f'{self.segmentation_value} min signal time split')
        print(f'{self.downsampling_value} sec data resampling')
        print(f'{self.num_class} classes classification')
        print(f'data balancing: {self.data_balancing}')
        print('-----')
        self.split_dataframe_lstm()

        valid_count = 0
        invalid_count = 0
        X = []
        y = []
        for arr in self.dataset:
            temp_X = []
            temp_y = []
            for i in range(len(arr)):
                temp_X.append(arr[i][0])
                temp_y.append(arr[i][1])
                invalid_count+=arr[i][1].count(0)
                valid_count+=arr[i][1].count(1)
            X.append(temp_X)
            y.append(temp_y)

        X = np.array(X, dtype=object)    # X data instances
        y_temp = []
        for labels in y:
            # y_temp.append(to_categorical(to_categorical(np.array(labels))))    # converting label list into categorical values
            y_temp.append(np.array(labels))
        y = np.array(y_temp, dtype=object)

        save_dir = os.path.dirname(os.path.abspath('util.py')) + f'/training/data/samples/{self.log_time}'
        os.mkdir(save_dir)

        # X numpy array resampling to fit model training
        # X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

        # splitting the data into testing and training sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        end_time = time.time()    # end timer variable used for the calculation of the total execution time 

        df_info_file = open(f'{save_dir}/info.txt', 'a')
        df_info_file.write(f'Is balanced = {self.is_balanced} \n')
        df_info_file.write(f'Segmentation value = {self.segmentation_value} \n')
        df_info_file.write(f'Downsampling value = {self.downsampling_value} \n')
        df_info_file.write(f'Signal resolution = {1/self.downsampling_value} \n')
        df_info_file.write('--- \n')
        df_info_file.write(f'% valid data point = {valid_count/(valid_count+invalid_count)} \n')
        df_info_file.write(f'% invalid data point = {invalid_count/(valid_count+invalid_count)} \n')
        df_info_file.write('--- \n')
        df_info_file.write(f'X train shape = {X_train.shape} \n')
        df_info_file.write(f'X test shape = {X_test.shape} \n')
        df_info_file.write(f'y train shape = {y_train.shape} \n')
        df_info_file.write(f'y test shape = {y_test.shape} \n')
        df_info_file.write('--- \n')
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
