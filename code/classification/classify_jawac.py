import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import mne
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from keras.models import load_model

from pathlib import Path
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')

from util import datetime_conversion, f1, analysis_cutting, is_valid, block_print, enable_print, hours_conversion, signal_quality, get_value_in_line


def analysis_classification(edf, 
                            model_name_task1='LSTM',
                            model_name_task2='ResNet', 
                            model_path_task1='',
                            model_path_task2='',
                            show_graph=False, 
                            plt_save_path='', 
                            stop_print=False):
    """
    Primary function for classifying a jawac patient signal

    Parameters:

    -edf (--edf): edf file path containing time series data
    -model_name_task1 (--model_name_task1): learning model architecture for valid/invalid task, either MLP, CNN, ResNet or LSTM
    -model_name_task2 (--model_name_task2): learning model architecture for awake/sleep task, either MLP, CNN, ResNet or LSTM
    -model_path_task1 (model_path_task1): corresponds to the path of the first task model weights
    -model_path_task2 (model_path_task2): corresponds to the path of the second task model weights
    -show_graph (--show_graph): true to show the output graph, false to skip it and only use the output dictionary
    -plt_save_path (--plt_save_path): path to save a .png copy file of the output plot if desired
    -stop_print (--stop_print): true to block dictionary output print in terminal

    Returns:

    -dictionary: dictionary containing information computed on the jawac signal
        dictionary keys:    -'model_path_task1'
                            -'model_path_task2'
                            -'total_hours'
                            -'percentage_sleep'
                            -'percentage_awake'
                            -'percentage_invalid'
                            -'hours_sleep'
                            -'hours_awake'
                            -'hours_invalid'
                            -'total_signal_quality'
                            -'new_bound_start'
                            -'new_bound_end'
                            -'total_hours_new_bounds'
                            -'hours_sleep_new_bounds'
                            -'percentage_sleep_new_bounds'
                            -'is_valid'
                            -'df_jawac'
                            -'plot'
    """

    if stop_print:
        block_print()

    start = time.time()    # start timer variable used for the calculation of the total execution time 

    raw_data = mne.io.read_raw_edf(edf)    # edf file reading

    model_name_task1 = model_name_task1.lower()

    if model_path_task1 == '':
        saved_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task1/{model_name_task1}'
        best_dir = f'{saved_dir}/best-1.1'
        model_list = os.listdir(best_dir)

        print('----- MODEL CHOICE -----')
        for i in range(len(model_list)):
            print(f'({i+1}) {model_list[i]}')
        model_num = int(input('model number: '))
        if model_num > 0 and model_num <= len(model_list):
            model_path_task1 = f'{best_dir}/{model_list[model_num-1]}/best/model-best.h5'
            info_path_task1 = f'{best_dir}/{model_list[model_num-1]}/info.txt'
        else:
            most_recent_folder_path = sorted(Path(best_dir).iterdir(), key=os.path.getmtime)[::-1]
            most_recent_folder_path = [name for name in most_recent_folder_path if not (str(name).split('/')[-1]).startswith('.')]
            model_path_task1 = str(most_recent_folder_path[0]) + '/best/model-best.h5'
            info_path_task1 = str(most_recent_folder_path[0]) + '/info.txt'
        print('------------------------')
    else:
        info_path_task1 = os.path.normpath(model_path_task1).split(os.sep)[:-2]
        info_path_task1.append('info.txt')
        info_path_task1 = '/'.join(info_path_task1)
        info_path_task1 = os.path.normpath(info_path_task1)
        model_path_task1 = os.path.normpath(model_path_task1)

    info_file_task1 = open(info_path_task1)
    lines = info_file_task1.readlines()
    lines = lines[3:13]
    
    segmentation_value = float(get_value_in_line(lines[0]))    # window segmentation value in second
    downsampling_value = float(get_value_in_line(lines[1]))    # signal downsampling value in second
    batch_size = int(get_value_in_line(lines[3]))    # batch size value for prediction
    standard_scale = bool(int(get_value_in_line(lines[4])))    # boolean value for the standardizatin of the data state
    stateful = bool(int(get_value_in_line(lines[5])))    # boolean value for the stateful model state
    sliding_window = bool(int(get_value_in_line(lines[6])))    # boolean value for the sliding window state
    center_of_interest = int(get_value_in_line(lines[7]))    # center of interest size in seconds for the sliding window
    full_sequence = bool(int(get_value_in_line(lines[8])))    # boolean value to feed the entire sequence without dividing it into multiple windows
    return_sequences = bool(int(get_value_in_line(lines[9])))    # boolean value to return the state of each data point in the full sequence for the LSTM model

    data, times = raw_data[:]    # edf file data extraction
    times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])    # conversion to usable date dtype 
    df_jawac = pd.DataFrame()    # dataframe creation to store time series data 
    df_jawac.insert(0, 'times', times)
    df_jawac.insert(1, 'data', data[0])
    df_jawac = df_jawac.resample(str(downsampling_value) + 'S', on='times').median()['data'].to_frame(name='data')
    df_jawac.reset_index(inplace=True)
    data = df_jawac.data.tolist()

    if standard_scale:
        scaler = StandardScaler()
        temp = scaler.fit_transform(np.reshape(data, (-1, 1)))
        data = np.reshape(temp, (len(temp))).tolist()

    start_class = time.time()    # start variable used for the calculation of the final classification time 

    print()
    print('Task 1...')
    print()

    timer_task1 = time.time()

    # this bloc of code divides the given time series into windows of 'size' number of data corresponding to the segmentation value in second
    if not stateful:
        size = int((1 / downsampling_value) * segmentation_value)
        if not sliding_window:
            X_test_seq = np.array([(df_jawac.loc[i:i + (size-1), :].data).to_numpy() for i in range(0, len(df_jawac), size)], dtype=object)
        else:
            size = int(segmentation_value)   # in sec
            X_test_seq = np.array([(df_jawac.loc[i:i + (size-1), :].data).to_numpy() for i in range(0, len(df_jawac), center_of_interest)], dtype=object)
        X_test_seq_temp = []
        for arr in X_test_seq:
            if not return_sequences and model_name_task1 != 'resnet':
                arr = np.append(arr, pd.Series([np.var(arr)*1000]))
            X_test_seq_temp.append(arr)
        X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq_temp, padding='post', dtype='float64')
        X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))

    # model loader
    if os.path.exists(model_path_task1):
        model = load_model(model_path_task1, compile=True, custom_objects={'f1': f1})
    else:
        raise Exception(model_path_task1, '-> model path does not exist')

    classes = []    # classes list holds the predicted labels
    threshold = 0.6    # above this threshold, the model invalid prediction are kept, otherwise considered as valid
    step_size = int((1 / downsampling_value) * segmentation_value)
    if not stateful:
        predictions = model.predict(X_test_seq_pad)    # model.predict classifies the X data by predicting the y labels
        # loop that runs through the list of model predictions to keep the highest predicted probability values
        for item in predictions[:-1]:
            if not return_sequences:
                idx = np.argmax(item)
                if idx == 0:
                    if item[idx] > threshold:
                        classes.append((idx, item[idx]))
                    else:
                        classes.append((1, 0.5))
                else:
                    classes.append((idx, item[idx]))
            else:
                labels = np.argmax(item, axis=1)
                if len(classes) == 0 and sliding_window:
                    classes.append(labels[:int((segmentation_value//2)-(center_of_interest//2))])
                if sliding_window:
                    for i in range(int((segmentation_value//2)-(center_of_interest//2)), int(len(labels) - (segmentation_value//2)+(center_of_interest//2)), 1):
                        if labels[i] == 0:
                            if item[i][labels[i]] > threshold:
                                classes.append((labels[i], item[i][labels[i]]))
                            else:
                                classes.append((1, 0.5))
                        else:
                            classes.append((labels[i], item[i][labels[i]]))
                else:
                    for i in range(len(labels)):
                        if labels[i] == 0:
                            if item[i][labels[i]] > threshold:
                                classes.append((labels[i], item[i][labels[i]]))
                            else:
                                classes.append((1, 0.5))
                        else:
                            classes.append((labels[i], item[i][labels[i]]))
        if return_sequences and sliding_window:
            classes.append(labels[int((segmentation_value//2)+(center_of_interest//2)):])
        else:
            classes.append((0, 0.5))
    else:
        X = np.reshape(data[0:len(data)-round(((len(data)/(step_size*batch_size))%1)*(step_size*batch_size))], ((len(data)-round(((len(data)/(step_size*batch_size))%1)*(step_size*batch_size)))//step_size, step_size, 1))
        predictions = model.predict(X, batch_size=batch_size)
        classes = []
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                idx = np.argmax(predictions[i][j])
                pred_label = idx
                proba = predictions[i][j][idx]
                if pred_label == 0:
                    if proba > threshold:
                        classes.append((pred_label, proba))
                    else:
                        classes.append((1, 0.5))
                else:
                    classes.append((pred_label, proba))

    df_jawac['label'] = [1 for n in range(len(df_jawac))]
    df_jawac['proba'] = [0.5 for n in range(len(df_jawac))]

    for i in range(len(classes)):
        if sliding_window:
            if not return_sequences:
                mask = df_jawac.index[(step_size//2)+(i*center_of_interest)-(center_of_interest//2):(step_size//2)+(i*center_of_interest)+(center_of_interest//2)]
            else:
                mask = df_jawac.index[i]
        else:
            if not return_sequences:
                mask = df_jawac.index[i*step_size:i*step_size+step_size]
            else:
                mask = df_jawac.index[i]
        if classes[i][0] == 0:
            df_jawac.loc[mask, ['label']] = 0
        df_jawac.loc[mask, ['proba']] = classes[i][1]

    min_gap_time = int((60 / downsampling_value) * 1)   # in minutes
    previous_label = df_jawac.label[0]
    idx = []
    for i in range(len(df_jawac.label)):
        idx.append(i)
        if df_jawac.label[i] != previous_label:
            if len(idx) <= min_gap_time:
                if previous_label == 0:
                    df_jawac.loc[idx, 'label'] = 1
                elif previous_label == 1:
                    df_jawac.loc[idx, 'label'] = 0
                df_jawac.loc[idx, 'proba'] = 0.5
            else:
                idx = []
                idx.append(i)
        previous_label = df_jawac.label[i]

    if 1 in df_jawac['label'].value_counts().keys():
        valid_total = df_jawac['label'].value_counts()[1]    # counter for the total number of valid regions found
    else:
        valid_total = 0
    if 0 in df_jawac['label'].value_counts().keys():
        invalid_total = df_jawac['label'].value_counts()[0]    # counter for the total number of invalid regions found
    else:
        invalid_total = 0

    total_valid_rate = valid_total / len(df_jawac)
    total_invalid_rate = invalid_total / len(df_jawac)

    valid_percentage = total_valid_rate * 100
    invalid_percentage = total_invalid_rate * 100
    valid_hours = (valid_total * (1/downsampling_value)) / 3600
    invalid_hours = (invalid_total * (1/downsampling_value)) / 3600

    print('--------')

    print('valid percentage:', round(valid_percentage, 2), '%')
    print('invalid percentage:', round(invalid_percentage, 2), '%')

    print('--------')

    print('valid hours:', hours_conversion(valid_hours))
    print('invalid hours:', hours_conversion(invalid_hours))

    print('--------')

    print()
    print(f'---> Task 1 done in {round((time.time() - timer_task1), 2)} sec!')
    print()
    print('Task 2...')
    print()

    model_name_task2 = model_name_task2.lower()

    if model_path_task2 == '':
        saved_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task1/{model_name_task2}'
        best_dir = f'{saved_dir}/best-1.1'
        model_list = os.listdir(best_dir)

        print('----- MODEL CHOICE -----')
        for i in range(len(model_list)):
            print(f'({i+1}) {model_list[i]}')
        model_num = int(input('model number: '))
        if model_num > 0 and model_num <= len(model_list):
            model_path_task2 = f'{best_dir}/{model_list[model_num-1]}/best/model-best.h5'
            info_path_task2 = f'{best_dir}/{model_list[model_num-1]}/info.txt'
        else:
            most_recent_folder_path = sorted(Path(best_dir).iterdir(), key=os.path.getmtime)[::-1]
            most_recent_folder_path = [name for name in most_recent_folder_path if not (str(name).split('/')[-1]).startswith('.')]
            model_path_task2 = str(most_recent_folder_path[0]) + '/best/model-best.h5'
            info_path_task2 = str(most_recent_folder_path[0]) + '/info.txt'
        print('------------------------')
    else:
        info_path_task2 = os.path.normpath(model_path_task2).split(os.sep)[:-2]
        info_path_task2.append('info.txt')
        info_path_task2 = '/'.join(info_path_task2)
        info_path_task2 = os.path.normpath(info_path_task2)
        model_path_task2 = os.path.normpath(model_path_task2)

    info_file_task2 = open(info_path_task2)
    lines = info_file_task2.readlines()
    lines = lines[3:13]

    timer_task2 = time.time()
    
    segmentation_value = float(get_value_in_line(lines[0]))    # window segmentation value in second
    downsampling_value = float(get_value_in_line(lines[1]))    # signal downsampling value in second
    batch_size = int(get_value_in_line(lines[3]))    # batch size value for prediction
    standard_scale = bool(int(get_value_in_line(lines[4])))    # boolean value for the standardizatin of the data state
    stateful = bool(int(get_value_in_line(lines[5])))    # boolean value for the stateful model state
    sliding_window = bool(int(get_value_in_line(lines[6])))    # boolean value for the sliding window state
    center_of_interest = int(get_value_in_line(lines[7]))    # center of interest size in seconds for the sliding window
    full_sequence = bool(int(get_value_in_line(lines[8])))    # boolean value to feed the entire sequence without dividing it into multiple windows
    return_sequences = bool(int(get_value_in_line(lines[9])))    # boolean value to return the state of each data point in the full sequence for the LSTM model

    # model loader
    if os.path.exists(model_path_task2):
        model = load_model(model_path_task2, compile=True, custom_objects={'f1': f1})
    else:
        raise Exception(model_path_task2, '-> model path does not exist')

    data = df_jawac[df_jawac['label']==1].data.tolist()

    # this bloc of code divides the given time series into windows of 'size' number of data corresponding to the segmentation value in second
    if not stateful:
        size = int((1 / downsampling_value) * segmentation_value)
        if not sliding_window:
            X_test_seq = np.array([(data[i:i + size]) for i in range(0, len(data), size)], dtype=object)
        else:
            size = int(segmentation_value)   # in sec
            X_test_seq = np.array([(data[i:i + size]) for i in range(0, len(data), center_of_interest)], dtype=object)
        X_test_seq_temp = []
        for arr in X_test_seq:
            # arr = np.append(arr, pd.Series([np.var(arr)*1000]))
            X_test_seq_temp.append(arr)
        X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq_temp, padding='post', dtype='float64')
        X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))

    threshold = 0.5
    classes = []
    step_size = int((1 / downsampling_value) * segmentation_value)
    predictions = []
    if not stateful:
        predictions = model.predict(X_test_seq_pad)    # model.predict classifies the X data by predicting the y labels
        # loop that runs through the list of model predictions to keep the highest predicted probability values
        for item in predictions[:-1]:
            if not return_sequences:
                idx = np.argmax(item)
                if idx == 0:
                    if item[idx] > threshold:
                        classes.append((idx, item[idx]))
                    else:
                        classes.append((1, 0.5))
                else:
                    classes.append((idx, item[idx]))
            else:
                labels = np.argmax(item, axis=1)
                if len(classes) == 0 and sliding_window:
                    classes.append(labels[:int((segmentation_value//2)-(center_of_interest//2))])
                if sliding_window:
                    for i in range(int((segmentation_value//2)-(center_of_interest//2)), int(len(labels) - (segmentation_value//2)+(center_of_interest//2)), 1):
                        if labels[i] == 0:
                            if item[i][labels[i]] > threshold:
                                classes.append((labels[i], item[i][labels[i]]))
                            else:
                                classes.append((1, 0.5))
                        else:
                            classes.append((labels[i], item[i][labels[i]]))
                else:
                    for i in range(len(classes)):
                        if labels[i] == 0:
                            if item[i][labels[i]] > threshold:
                                classes.append((labels[i], item[i][labels[i]]))
                            else:
                                classes.append((1, 0.5))
                        else:
                            classes.append((labels[i], item[i][labels[i]]))
        if return_sequences and sliding_window:
            classes.append(labels[int((segmentation_value//2)+(center_of_interest//2)):])
    else:
        X = np.reshape(data[0:len(data)-int(((len(data)/(step_size*batch_size))%1)*(step_size*batch_size))], ((len(data)-int(((len(data)/(step_size*batch_size))%1)*(step_size*batch_size)))//step_size, step_size, 1))
        predictions = model.predict(X, batch_size=batch_size)
        classes = []
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                idx = np.argmax(predictions[i][j])
                pred_label = idx
                proba = predictions[i][j][idx]
                if pred_label == 0:
                    if proba > threshold:
                        classes.append((pred_label, proba))
                    else:
                        classes.append((1, 0.5))
                else:
                    classes.append((pred_label, proba))

    df_jawac_only_valid = df_jawac[df_jawac['label']==1]

    for i in range(len(classes)):
        if sliding_window:
            if not return_sequences:
                mask = df_jawac_only_valid.index[(step_size//2)+(i*center_of_interest)-(center_of_interest//2):(step_size//2)+(i*center_of_interest)+(center_of_interest//2)]
            else:
                mask = df_jawac_only_valid.index[i]
        else:
            if not return_sequences:
                mask = df_jawac_only_valid.index[i*step_size:i*step_size+step_size]
            else:
                mask = df_jawac_only_valid.index[i]
        if classes[i][0] == 0:
            df_jawac.loc[mask, ['label']] = 2
        df_jawac.loc[mask, ['proba']] = classes[i][1]

    min_gap_time = int((60 / downsampling_value) * 3)   # in minutes
    previous_label = df_jawac.label[0]
    idx = []
    for i in range(len(df_jawac.label)):
        idx.append(i)
        if df_jawac.label[i] != previous_label:
            if len(idx) <= min_gap_time:
                if previous_label == 2:
                    df_jawac.loc[idx, 'label'] = 1
                df_jawac.loc[idx, 'proba'] = 0.5
            else:
                idx = []
                idx.append(i)
        previous_label = df_jawac.label[i]

    min_gap_time = int((60 / downsampling_value) * 3)   # in minutes
    previous_label = df_jawac.label[0]
    idx = []
    for i in range(len(df_jawac.label)):
        idx.append(i)
        if df_jawac.label[i] != previous_label:
            if len(idx) <= min_gap_time:
                if previous_label == 1:
                    df_jawac.loc[idx, 'label'] = 2
                df_jawac.loc[idx, 'proba'] = 0.5
            else:
                idx = []
                idx.append(i)
        previous_label = df_jawac.label[i]

    print(f'---> Task 2 done in {round((time.time() - timer_task2), 2)} sec!')
    print()

    df_label = pd.DataFrame(columns=['start', 'end', 'label', 'proba', 'data_num'])
    saved_label = df_jawac.iloc[0].label
    saved_start = df_jawac.iloc[0].times
    proba = []
    for idx, row in df_jawac.iterrows():
        if len(row) > 0:
            proba.append(row.proba)
            if row.label != saved_label:
                temp = pd.DataFrame(data = [[saved_start, row.times, saved_label, np.mean(proba), len(proba)]], columns=['start', 'end', 'label', 'proba', 'data_num'])
                df_label = pd.concat([df_label, temp], ignore_index=True)
                saved_label = row.label
                saved_start = row.times
                proba = []
    if len(proba) > 0:
        temp = pd.DataFrame(data = [[saved_start, df_jawac.times.tolist()[-1], saved_label, np.mean(proba), len(proba)]], columns=['start', 'end', 'label', 'proba', 'data_num'])
        df_label = pd.concat([df_label, temp], ignore_index=True)

    # call of the function that proposes new bounds for the breakdown of the analysis for the diagnosis
    df_label, sleep_hours, sleep_rate, new_start, new_end = analysis_cutting(df=df_label, analysis_start=df_jawac.index[0], analysis_end=df_jawac.index[-1], downsampling_value=downsampling_value, threshold=0.98)

    print('new start analysis time:', new_start)
    print('new end analysis time:', new_end)
    if new_start is not None and new_end is not None:
        duration = (new_end - new_start)
        print('new analysis duration:', duration)
    else:
        duration = timedelta(seconds=0)
        print('analysis duration: Unknown')
    print('sleep time in new bounds:', hours_conversion(sleep_hours))
    print('sleep rate in new bounds:', round((sleep_rate * 100), 2), '%')

    print('--------')
    
    print('total signal quality score (/100):', round((signal_quality(df_label) * 100), 2))

    print('--------')

    print('is analysis valid:', is_valid(times[-1] - times[0], sleep_hours))

    print('--------')

    if 0 in df_jawac['label'].value_counts().keys():
        invalid_count = df_jawac['label'].value_counts()[0]
    else:
        invalid_count = 0
    if 1 in df_jawac['label'].value_counts().keys():
        valid_count = df_jawac['label'].value_counts()[1]
    else:
        valid_count = 0
    if 2 in df_jawac['label'].value_counts().keys():
        awake_count = df_jawac['label'].value_counts()[2]
    else:
        awake_count = 0

    print(f'invalid count: {invalid_count} --> {round((invalid_count/(invalid_count+valid_count+awake_count)*100), 2)} %')
    print(f'valid count: {valid_count} --> {round((valid_count/(invalid_count+valid_count+awake_count)*100), 2)} %')
    print(f'awake count: {awake_count} --> {round((awake_count/(invalid_count+valid_count+awake_count)*100), 2)} %')

    print('--------')

    total_invalid_rate = invalid_count/len(df_jawac)
    total_valid_rate = valid_count/len(df_jawac)
    total_awake_rate = awake_count/len(df_jawac)

    # creation of the output dictionary with computed information about the signal
    dictionary = {'model_path_task1': model_path_task1, 
                  'model_path_task2': model_path_task2, 
                  'total_hours': round(((times[-1] - times[0]).total_seconds() / 3600.0), 2), 
                  'percentage_sleep': round((total_valid_rate * 100), 2), 
                  'percentage_awake': round((total_awake_rate * 100), 2), 
                  'percentage_invalid': round((total_invalid_rate * 100), 2), 
                  'hours_sleep': round(((valid_count * downsampling_value) / 3600), 2), 
                  'hours_awake': round(((awake_count * downsampling_value) / 3660), 2), 
                  'hours_invalid': round(((invalid_count * downsampling_value) / 3600), 2), 
                  'total_signal_quality': round((signal_quality(df_label) * 100), 2), 
                  'new_bound_start': new_start, 
                  'new_bound_end': new_end, 
                  'total_hours_new_bounds': round((duration.total_seconds() / 3600.0), 2), 
                  'hours_sleep_new_bounds': round(sleep_hours, 2), 
                  'percentage_sleep_new_bounds': round(sleep_rate * 100, 2), 
                  'df_jawac': df_jawac,
                  'is_valid': is_valid(times[-1] - times[0], sleep_hours)}

    end = time.time()    # end timer variable used for the calculation of the total execution time 

    print('classification execution time =', round((end - start_class), 2), 'sec')
    print('total execution time =', round((end - start), 2), 'sec')

    fig = go.Figure()

    df_jawac.set_index('times', inplace=True)
    df_jawac.proba = df_jawac.proba.round(3)

    fig.add_trace(
        go.Scatter(x=df_jawac.index.tolist(), 
                   y=df_jawac.data.tolist(), 
                   showlegend=False,
                   customdata = df_jawac.proba.tolist(),
                   hovertemplate='<extra><br>proba: %{customdata}</extra>',
                   hoverinfo='skip',
                   line=dict(
                       color='rgb(57, 119, 175)',
                       width=2)),
    )

    for idx, row in df_label.iterrows():
        if row.label == 0:
            fig.add_vrect(x0=row.start, x1=row.end,
                          fillcolor='red', 
                          opacity=0.2*row.proba,
                          line_width=0)
        elif row.label == 1:
            fig.add_vrect(x0=row.start, x1=row.end,
                          fillcolor='green', 
                          opacity=0.2*row.proba,
                          line_width=0)
        elif row.label == 2:
            fig.add_vrect(x0=row.start, x1=row.end,
                          fillcolor='blue', 
                          opacity=0.2*row.proba,
                          line_width=0)

    fig.add_hline(y=0, line_color='red', line_width=1)

    fig.add_vline(x=new_start, line_dash="dash", line_color='black', line_width=2)
    fig.add_vline(x=new_end, line_dash="dash", line_color='black', line_width=2)

    fig.update_xaxes(title_text="time")
    fig.update_yaxes(title_text="opening (mm)")

    legend = 'red(invalid), green(valid/sleep), blue(awake)'

    title = ''    # value of the chart title, containing the name of the analysis
    for c in reversed(edf[:-4]):
        if c != '/':
            title += c
        else:
            break
    title = title[::-1]

    fig.add_annotation(text=f'model task 1: {model_name_task1} - model task 2: {model_name_task2} - total time: {hours_conversion(dictionary["total_hours"])} - hours sleep: {hours_conversion(dictionary["hours_sleep"])} - new bounds time: {hours_conversion(dictionary["total_hours_new_bounds"])} - new bounds sleep time: {hours_conversion(dictionary["hours_sleep_new_bounds"])} - total signal quality: {dictionary["total_signal_quality"]} - valid: {dictionary["is_valid"]}',
                       xref='paper', yref='paper',
                       x=0.3, y=1.045, showarrow=False)

    fig.update_layout(title=go.layout.Title(
                            text=f'{title} <br><sup>Legend: {legend}</sup>',
                            xref='paper',
                            x=0))

    fig.layout.width = None
    fig.layout.height = None

    if plt_save_path != '':
        fig.write_image(f'{plt_save_path}/{model_name_task1}/{threshold}/output_{title}.png')
        fig.write_image(f'{plt_save_path}/{model_name_task2}/{threshold}/output_{title}.png')

    dictionary['plot'] = fig
    if show_graph:
        fig.show()

    enable_print()

    return dictionary


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default='', help='edf file path containing time series data')
    parser.add_argument('--model_name_task1', type=str, default='LSTM', help='learning model architecture for valid/invalid task - either MLP, CNN, ResNet or LSTM')
    parser.add_argument('--model_name_task2', type=str, default='ResNet', help='learning model architecture for awake/sleep task - either MLP, CNN, ResNet or LSTM')
    parser.add_argument('--model_path_task1', type=str, default='', help='corresponds to the path of the first task model weights')
    parser.add_argument('--model_path_task2', type=str, default='', help='corresponds to the path of the second task model weights')
    parser.add_argument('--show_graph', dest='show_graph', action='store_true', help='invoke to show the output graph')
    parser.add_argument('--plt_save_path', type=str, default='', help='path to save a .png copy file of the output plot if desired')
    parser.add_argument('--stop_print', dest='stop_print', action='store_true', help='invoke to block dictionary output print in terminal')
    parser.set_defaults(show_graph=False)
    parser.set_defaults(stop_print=False)
    return parser.parse_args()


def main(p):
    out_dic = analysis_classification(**vars(p))


if __name__ == '__main__':
    # statement to avoid useless warnings during training
    # export TF_CPP_MIN_LOG_LEVEL=3
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    # Test cmd lines
    # python code/classification/classify_jawac.py --show_graph --model_name_task1 'LSTM' --model_path_task1 'models\task1\lstm\best-1.0\stl-lstm-task1-1.0#2\best\model-best.h5' --model_name_task2 'ResNet' --model_path_task2 'models\task2\resnet\best-1.0\resnet-sw-task2#2\best\model-best.h5' --edf 'data\task2\jawrhin\analysis\03\03.edf'


    opt = parse_opt()
    main(p=opt)
