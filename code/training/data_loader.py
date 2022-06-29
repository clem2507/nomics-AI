import os
import mne
import sys
import time
import shutil
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')

from util import extract_data_from_line, datetime_conversion, block_print, enable_print, occurrence_count


class DataLoader:
    """
    Class used to preprocess the created dataframes

    Attributes:

    -analysis_directory: directory path containing the analyses to create the dataframes
    -segmentation_value: window segmentation value in second
    -downsampling_value: signal downsampling value in second
    -data_balancing: true is balanced data is needed, false otherwise
    -log_time: save the time when the computation starts for the directory name
    -sliding_window: true to use sliding window with a small center portion of interest
    -center_of_interest: center of interest size in seconds for the sliding window
    -task: corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep
    """

    def __init__(self, 
                 analysis_directory='', 
                 model_name='ResNet',
                 segmentation_value=60, 
                 downsampling_value=1, 
                 data_balancing=True, 
                 log_time='', 
                 sliding_window=False, 
                 center_of_interest=10, 
                 task=1):

        self.directory = analysis_directory
        self.model_name = model_name
        self.dfs_directory = f'{analysis_directory}_edf_dfs'
        self.segmentation_value = float(segmentation_value)
        self.downsampling_value = float(downsampling_value)
        self.data_balancing = data_balancing
        self.log_time = log_time
        self.sliding_window = sliding_window
        self.center_of_interest = center_of_interest
        self.task = task
        self.jawac_df_list = []
        self.mk3_df_list = []
        self.dataset = []

    
    def create_dataframes(self):
        """
        Function that update, creates and saves the dataframes based on the analysis files
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
                    lines = lines[5:]    # log file information
                    if len(lines) > 0:
                        if (lines[0].split(';')[-1].strip()) == 'START':
                            lines.pop(0)
                    if len(lines) > 0:
                        if (lines[0].split(';')[-1].strip()) == 'START':
                            lines.pop(0)
                    if len(lines) > 0:
                        if (lines[0].split(';')[-1].strip()) == 'STOP':
                            lines.pop(0)
                    if len(lines) > 0:
                        if (lines[0].split(';')[-1].strip()) == 'STOP':
                            lines.pop(0)
                    col_names = ['start', 'end', 'signal_num', 'label']
                    df_mk3 = pd.DataFrame(columns=col_names)    # dataframe with label data created
                    for line in lines:
                        if len(line) > 0:
                            temp = pd.DataFrame(extract_data_from_line(line), columns=df_mk3.columns)
                            df_mk3 = pd.concat([df_mk3, temp], ignore_index=True)

                    # dataframe saving
                    df_mk3.to_pickle(f'{self.dfs_directory}/{dir_names[i]}/{dir_names[i]}_mk3.pkl')

                    block_print()
                    raw_data = mne.io.read_raw_edf(edf_file)    # edf file reading
                    enable_print()

                    df_jawac = pd.DataFrame()    # dataframe creation to store time series data

                    jawac_signal_idx = raw_data.__dict__['_raw_extras'][0]['ch_names'].index('Jawac')
                    data = raw_data[jawac_signal_idx][0][0]
                    times = raw_data[0][1]
                    times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])    # conversion to usable date dtype 
                    df_jawac.insert(0, 'times', times)
                    df_jawac.insert(1, 'data', data)
                    if 'Hypno' in raw_data.__dict__['_raw_extras'][0]['ch_names']:
                        hypno_signal_idx = raw_data.__dict__['_raw_extras'][0]['ch_names'].index('Hypno')
                        df_jawac.insert(1, 'hypno', raw_data[hypno_signal_idx][0][0])
                    df_jawac = df_jawac.resample('0.1S', on='times').median()

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


    def split_dataframe(self):
        """
        Method used to segment the dataframes into one common dataset
        """

        dir_names = sorted(os.listdir(self.dfs_directory))
        for i in tqdm(range(len(dir_names))):
            if not dir_names[i].startswith('.'):
                df_mk3 = pd.read_pickle(f'{self.dfs_directory}/{dir_names[i]}/{dir_names[i]}_mk3.pkl')
                df_jawac = pd.read_pickle(f'{self.dfs_directory}/{dir_names[i]}/{dir_names[i]}_jawac.pkl')
                df_jawac = df_jawac.resample(str(self.downsampling_value) + 'S').median()
                self.mk3_df_list.append(df_mk3)
                self.jawac_df_list.append(df_jawac)

        for i in tqdm(range(len(self.jawac_df_list))):
            self.jawac_df_list[i]['label'] = 1
            self.jawac_df_list[i].index = pd.to_datetime(self.jawac_df_list[i].index).tz_localize(None)
            for idx, row in self.mk3_df_list[i].iterrows():
                mask = (self.jawac_df_list[i].index >= row.start) & (self.jawac_df_list[i].index <= row.end)
                if self.task == 1:
                    self.jawac_df_list[i].loc[mask, ['label']] = 0
                elif self.task == 2:
                    if row.label == 'W':
                        self.jawac_df_list[i].loc[mask, ['label']] = 0
                    else:
                        self.jawac_df_list[i] = self.jawac_df_list[i][~mask]
                
        for i in tqdm(range(len(self.jawac_df_list))):
            self.dataset.append([self.jawac_df_list[i].data.tolist(), self.jawac_df_list[i].label.tolist()])


    def create_dataset(self):
        """
        Method that transforms the dataframe dataset into numpy array to better fit the learning model

        Returns:

        -X_train: x data instances for training the model
        -y_train: y label instances for training the model
        -X_test: x data instances for testing the model
        -y_test: y label instances for testing the model
        """

        start_time = time.time()    # start timer variable used for the calculation of the total execution time 

        print('-----')
        print(f'Task {self.task}')
        print(f'{self.downsampling_value} sec data resampling --> signal resolution: {1/self.downsampling_value} Hz')
        print(f'{self.segmentation_value} sec segmentation window')
        if self.sliding_window:
            print(f'center of interest value: {round(self.center_of_interest/60, 2)} min, {self.center_of_interest} sec')
        print(f'data balancing: {self.data_balancing}')
        print(f'sliding window: {self.sliding_window}')
        print('-----')

        self.split_dataframe()

        one_count = 0
        zero_count = 0
        X = []
        y = []
        max_length = int(self.segmentation_value * (1 / self.downsampling_value))
        for arr in self.dataset:
            if self.sliding_window:
                temp_X = [arr[0][i:i + max_length] for i in range(0, len(arr[0]), self.center_of_interest)][:-max_length//self.center_of_interest]
                temp_y = [arr[1][i:i + max_length] for i in range(0, len(arr[1]), self.center_of_interest)][:-max_length//self.center_of_interest]
            else:
                temp_X = [arr[0][i:i + max_length] for i in range(0, len(arr[0]), max_length)][:-1]
                temp_y = [arr[1][i:i + max_length] for i in range(0, len(arr[1]), max_length)][:-1]
            for i in range(len(temp_X)):
                if self.task == 1 and not self.model_name.lower() == 'resnet':
                    temp_X[i].append(np.var(temp_X[i])*1000)
                if occurrence_count(temp_y[i], threshold=0.95):
                    label = max(temp_y[i], key=temp_y[i].count)    # most common value in the data window
                else:
                    continue
                if label == 0:
                    zero_count += 1
                elif label == 1:
                    one_count += 1
                X.append(temp_X[i])
                y.append(label)

        if self.data_balancing:
            X = np.array(X)
            y = np.array(y)
            oversample = SMOTE()
            X, y = oversample.fit_resample(X.reshape(X.shape[0], -1), y)
            X = np.reshape(X, (len(X), len(X[0]), 1))
            y = to_categorical(y)
            zero_count = 0
            one_count = 0
            for label in y:
                if label[0] == 1:
                    zero_count += 1
                else:
                    one_count += 1
        else:
            X = np.reshape(X, (len(X), len(X[0]), 1))
            y = to_categorical(np.array(y))
            X, y = shuffle(X, y)

        # splitting the data into testing and training sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        end_time = time.time()    # end timer variable used for the calculation of the total execution time 

        print('----')
        print(f'% 1 data point = {round(100*(one_count/(one_count+zero_count)), 2)}')
        print(f'% 0 data point = {round(100*(zero_count/(one_count+zero_count)), 2)}')
        print('----')
        print(f'X train length = {len(X_train)}')
        print(f'X test length = {len(X_test)}')
        print(f'y train length = {len(y_train)}')
        print(f'y test length = {len(y_test)}')
        print('----')
        print(f'Total data loder computation time = {round(end_time-start_time, 2)} sec | {round((end_time-start_time)/60, 2)} min')
        print('----')

        return X_train, y_train, X_test, y_test
