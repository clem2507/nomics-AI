from enum import Flag
import os
import sys
import mne
import time
import math
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import load_model

from pathlib import Path
from datetime import timedelta
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')

from util import datetime_conversion, f1_m, analysis_cutting, is_valid, block_print, enable_print, hours_conversion, signal_quality, get_value_in_line, extract_data_from_line, string_datetime_conversion


def analysis_classification(edf, model, view_graph, plt_save_path):
    """
    Primary function for classifying an edf patient file

    Parameters:

    -edf (--edf): edf file path containing time series data
    -model (--model): learning model architecture, either CNN, LSTM or KNN
    -view_graph (--view_graph): true to show the output graph, false to skip it and only use the output dictionary
    -plt_save_path (--plt_save_path): path to save a .png copy file of the output plot if desired
    -task (--task): corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep

    Returns:

    -dictionary: dictionary containing information computed on the jawac signal
        dictionary keys:    -'model_path'
                            -'total_hours'
                            -'percentage_sleep'
                            -'percentage_awake'
                            -'percentage_invalid'
                            -'hours_sleep'
                            -'hours_awake'
                            -'hours_invalid
                            -'total_signal_quality'
                            -'new_bound_start'
                            -'new_bound_end'
                            -'total_hours_new_bounds'
                            -'hours_sleep_new_bounds'
                            -'percentage_sleep_new_bounds'
                            -'signal_quality_new_bounds'
                            -'is_valid'
                            -'plot'
    """

    # block_print()

    start = time.time()    # start timer variable used for the calculation of the total execution time 

    model_name = model.lower()
    saved_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task1/{model_name}'

    most_recent_folder_path = sorted(Path(saved_dir).iterdir(), key=os.path.getmtime)[::-1]
    most_recent_folder_path = [name for name in most_recent_folder_path if not (str(name).split('/')[-1]).startswith('.')]

    model_path = str(most_recent_folder_path[0]) + '/saved_model'
    info_path = str(most_recent_folder_path[0]) + '/info.txt'
    info_file = open(info_path)
    lines = info_file.readlines()
    lines = lines[3:9]
    
    segmentation_value = float(get_value_in_line(lines[0]))    # window segmentation value in minute
    downsampling_value = float(get_value_in_line(lines[1]))    # signal downsampling value in second
    batch_size = int(get_value_in_line(lines[3]))    # batch size value for prediction
    standard_scale = bool(int(get_value_in_line(lines[4])))    # boolean value for the standardizatin of the data
    stateful = bool(int(get_value_in_line(lines[5])))    # boolean value for the stateful model

    raw_data = mne.io.read_raw_edf(edf)    # edf file reading

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

    # this bloc of code divides the given time series into windows of 'size' number of data corresponding to the segmentation value in minute
    if not stateful:
        size = int((60 / downsampling_value) * segmentation_value)
        X_test_seq = np.array([(df_jawac.loc[i:i + size - 1, :].data).to_numpy() for i in range(0, len(df_jawac), size)], dtype=object)
        X_test_seq_temp = []
        for arr in X_test_seq:
            arr = np.append(arr, pd.Series([np.var(arr)*1000]))
            X_test_seq_temp.append(arr)
        X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq_temp, padding='post', dtype='float64')
        X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))

    # model loader
    if os.path.exists(model_path):
        if model_name in ['cnn', 'lstm']:
            model = load_model(model_path, compile=True, custom_objects={'f1_m': f1_m})
    else:
        raise Exception('model path does not exist')

    classes = []    # classes list holds the predicted labels
    threshold = 0.5    # above this threshold, the model invalid prediction are kept, otherwise considered as valid
    if not stateful:
        minutes_per_class = segmentation_value
        predictions = model.predict(X_test_seq_pad)    # model.predict classifies the X data by predicting the y labels
        # loop that runs through the list of model predictions to keep the highest predicted probability values
        for item in predictions[:-1]:
            idx = np.argmax(item)
            print((idx, item[idx]), end='')
            if model_name in ['cnn', 'lstm']:
                if idx == 0:
                    if item[idx] > threshold:
                        classes.append((idx, item[idx]))
                    else:
                        classes.append((1, 0.5))
                else:
                    classes.append((idx, item[idx]))
            else:
                classes.append((idx, 1))
    else:
        minutes_per_class = (batch_size * downsampling_value)/60
        for i in range(0, len(data), batch_size):
            if i+batch_size < len(data):
                y_pred, *r = model.predict_on_batch(np.reshape(data[i:i+batch_size], (batch_size, 1, 1)))
                label = round(y_pred[0][0])
                if label == 0:
                    if 1-y_pred[0][0] > threshold:
                        classes.append((label, 1-y_pred[0][0]))
                    else:
                        classes.append((1, 0.5))
                else:
                    classes.append((label, y_pred[0][0]))
        model.reset_states()
    classes.append((0, 0.5))
    print()

    minimum_valid_time = 3   # in minutes
    valid_threshold = math.ceil(minimum_valid_time/minutes_per_class)
    valid_idx = []
    for i in range(len(classes)):
        if classes[i][0] == 0:
            if len(valid_idx) > 0 and len(valid_idx) <= valid_threshold:
                for pos in valid_idx:
                    classes[pos] = (0, 0.5)
            valid_idx = []
        else:
            valid_idx.append(i)
    minimum_invalid_time = 1   # in minutes
    invalid_threshold = math.ceil(minimum_invalid_time/minutes_per_class)
    invalid_idx = []
    for i in range(1, len(classes)-1):
        if classes[i][0] == 0:
            invalid_idx.append(i)
        else:
            if len(invalid_idx) > 0 and len(invalid_idx) <= invalid_threshold:
                for pos in invalid_idx:
                    classes[pos] = (1, 0.5)
            invalid_idx = []

    valid_total = 0    # counter for the total number of valid regions found
    invalid_total = 0    # counter for the total number of invalid regions found
    for label in classes:
        if label[0] == 0:
            invalid_total += 1
        else:
            valid_total += 1
    total_valid_rate = valid_total / len(classes)
    total_invalid_rate = invalid_total / len(classes)

    valid_percentage = total_valid_rate * 100
    invalid_percentage = total_invalid_rate * 100
    valid_hours = (valid_total * minutes_per_class) / 60
    invalid_hours = (invalid_total * minutes_per_class) / 60

    print('--------')

    print('valid percentage:', round(valid_percentage, 2), '%')
    print('invalid percentage:', round(invalid_percentage, 2), '%')

    print('--------')

    print('valid hours:', hours_conversion(valid_hours))
    print('invalid hours:', hours_conversion(invalid_hours))

    print('--------')
    
    print('total signal quality', round((signal_quality(classes) * 100), 2))

    print('--------')

    if not is_valid(times[-1] - times[0], (valid_total * minutes_per_class) / 60):
        raise Exception('---> Analysis invalid')

    print('---> Task 1 done!')
    print('Task 2...')
    print()

    step_size = int((60 / downsampling_value) * segmentation_value)
    df_jawac['label'] = [1 for n in range(len(df_jawac))]
    for i in range(len(classes)):
        if classes[i][0] == 0:
            if (i*step_size)+step_size < len(df_jawac):
                df_jawac.loc[(i*step_size):(i*step_size)+step_size, 'label'] = 0

    saved_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task2/{model_name}'

    most_recent_folder_path = sorted(Path(saved_dir).iterdir(), key=os.path.getmtime)[::-1]
    most_recent_folder_path = [name for name in most_recent_folder_path if not (str(name).split('/')[-1]).startswith('.')]

    model_path = str(most_recent_folder_path[0]) + '/saved_model'
    info_path = str(most_recent_folder_path[0]) + '/info.txt'
    info_file = open(info_path)
    lines = info_file.readlines()
    lines = lines[3:9]
    
    segmentation_value = float(get_value_in_line(lines[0]))    # window segmentation value in minute
    downsampling_value = float(get_value_in_line(lines[1]))    # signal downsampling value in second
    batch_size = int(get_value_in_line(lines[3]))    # batch size value for prediction
    standard_scale = bool(int(get_value_in_line(lines[4])))    # boolean value for the standardizatin of the data
    stateful = bool(int(get_value_in_line(lines[5])))    # boolean value for the stateful model

    # model loader
    if os.path.exists(model_path):
        if model_name in ['cnn', 'lstm']:
            model = load_model(model_path, compile=True, custom_objects={'f1_m': f1_m})
    else:
        raise Exception('model path does not exist')

    X = df_jawac[df_jawac['label']!=0].data.tolist()

    # this bloc of code divides the given time series into windows of 'size' number of data corresponding to the segmentation value in minute
    if not stateful:
        size = int((60 / downsampling_value) * segmentation_value)
        X_test_seq = np.array([(X[i:i + size]) for i in range(0, len(X), size)], dtype=object)
        X_test_seq_temp = []
        for arr in X_test_seq:
            arr = np.append(arr, pd.Series([np.var(arr)*1000]))
            X_test_seq_temp.append(arr)
        X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq_temp, padding='post', dtype='float64')
        X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))

    threshold = 0.5
    minutes_per_class = (batch_size * downsampling_value)/60
    classes = []
    if not stateful:
        minutes_per_class = segmentation_value
        predictions = model.predict(X_test_seq_pad)    # model.predict classifies the X data by predicting the y labels
        # loop that runs through the list of model predictions to keep the highest predicted probability values
        step_size = int((60 / downsampling_value) * segmentation_value)
        for item in predictions[:-1]:
            idx = np.argmax(item)
            print((idx, item[idx]), end='')
            if model_name in ['cnn', 'lstm']:
                if idx == 0:
                    if item[idx] > threshold:
                        classes.append((idx, item[idx]))
                    else:
                        classes.append((1, 0.5))
                else:
                    classes.append((idx, item[idx]))
            else:
                classes.append((idx, 1))
    else:
        step_size = batch_size
        for i in range(0, len(X), batch_size):
            if i+batch_size < len(X):
                y_pred, *r = model.predict_on_batch(np.reshape(X[i:i+batch_size], (batch_size, 1, 1)))
                label = round(y_pred[0][0])
                if label == 0:
                    if 1-y_pred[0][0] > threshold:
                        classes.append((label, 1-y_pred[0][0]))
                    else:
                        classes.append((1, 0.5))
                else:
                    classes.append((label, y_pred[0][0]))
            model.reset_states()
    print()

    for i in range(len(classes)):
        if classes[i][0] == 0:
            if (i*step_size)+step_size < len(df_jawac):
                mask = df_jawac[df_jawac['label']==1].index[i*step_size:i*step_size+step_size]
                df_jawac.loc[mask, ['label']] = 2

    df_jawac.set_index('times', inplace=True)

    # call of the function that proposes new bounds for the breakdown of the analysis for the diagnosis - TODO - adapt this function to the new problem
    new_bounds_classes, valid_hours, valid_rate, new_start, new_end = analysis_cutting(classes=classes, analysis_start=df_jawac.index[0], analysis_end=df_jawac.index[-1], minutes_per_class=minutes_per_class, downsampling_value=downsampling_value, threshold=0.98)

    print('new start analysis time:', new_start)
    print('new end analysis time:', new_end)
    if new_start is not None and new_end is not None:
        duration = (new_end - new_start)
        print('new analysis duration:', duration)
    else:
        duration = timedelta(seconds=0)
        print('analysis duration: Unknown')
    print('valid time in new bounds:', hours_conversion(valid_hours))
    print('valid rate in new bounds:', round((valid_rate * 100), 2), '%')

    print('--------')
    
    print('signal quality in new bounds', round((signal_quality(new_bounds_classes) * 100), 2))

    print('--------')

    print('is analysis valid:', is_valid(times[-1] - times[0], valid_hours))

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

    print(invalid_count)
    print(valid_count)
    print(awake_count)

    total_invalid_rate = invalid_count/len(df_jawac)
    total_valid_rate = valid_count/len(df_jawac)
    total_awake_rate = awake_count/len(df_jawac)

    # creation of the output dictionary with computed information about the signal - TODO - redo the signal quality function
    dictionary = {'model_path': model_path, 'total_hours': round(((times[-1] - times[0]).total_seconds() / 3600.0), 2), 'percentage_sleep': round((total_valid_rate * 100), 2), 'percentage_awake': round((total_awake_rate * 100), 2), 'percentage_invalid': round((total_invalid_rate * 100), 2), 'hours_sleep': round(((valid_count * downsampling_value) / 3600), 2), 'hours_awake': round(((awake_count * downsampling_value) / 3660), 2), 'hours_invalid': round(((invalid_count * downsampling_value) / 3600), 2), 'total_signal_quality': round((signal_quality(classes) * 100), 2), 'new_bound_start': new_start, 'new_bound_end': new_end, 'total_hours_new_bounds': round((duration.total_seconds() / 3600.0), 2), 'hours_sleep_new_bounds': round(valid_hours, 2), 'percentage_sleep_new_bounds': round(valid_rate * 100, 2), 'signal_quality_new_bounds': round((signal_quality(new_bounds_classes) * 100), 2), 'is_valid': is_valid(times[-1] - times[0], valid_hours)}

    # graph
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    # plot of the time series values
    ax.plot(df_jawac.index.tolist(), data)
    ax.axhline(y=0, color='r', linewidth=1)
    if new_start is not None and new_end is not None:
        ax.axvline(x=new_start, color='k', linewidth=2, linestyle='--')
        ax.axvline(x=new_end, color='k', linewidth=2, linestyle='--')
    title = ''    # value of the chart title, containing the name of the analysis
    for c in reversed(edf[:-4]):
        if c != '/':
            title += c
        else:
            break
    title = title[::-1]
    ax.set(xlabel='time', ylabel='opening (mm)', title=f'Jawac Signal - {title}')
    ax.grid()

    df_jawac.reset_index(inplace=True)
    df_label = pd.DataFrame(columns=['start', 'end', 'label'])
    saved_label = df_jawac.iloc[0].label
    saved_start = df_jawac.iloc[0].times
    for idx, row in df_jawac.iterrows():
        if len(row) > 0:
            if row.label != saved_label:
                a_row = {'start': saved_start, 'end': row.times, 'label': saved_label}
                df_label = df_label.append(a_row, ignore_index=True)
                saved_label = row.label
                saved_start = row.times
    a_row = {'start': saved_start, 'end': df_jawac.times.tolist()[-1], 'label': saved_label}
    df_label = df_label.append(a_row, ignore_index=True)

    curr_time = raw_data.__dict__['info']['meas_date']    # starting time of the analysis -- Not needed anymore
    # graph background color based on signal classified label
    for idx, row in df_label.iterrows():
        if row.label == 0:
            plt.axvspan(row.start, row.end, facecolor='r', alpha=0.3)
        elif row.label == 1:
            plt.axvspan(row.start, row.end, facecolor='g', alpha=0.3)
        elif row.label == 2:
            plt.axvspan(row.start, row.end, facecolor='b', alpha=0.3)

    # legend
    legend_elements = [Patch(facecolor='r', edgecolor='w', label='invalid', alpha=0.25),
                    Patch(facecolor='g', edgecolor='w', label='sleep', alpha=0.25),
                    Patch(facecolor='b', edgecolor='w', label='awake', alpha=0.25),
                    Line2D([0], [0], linewidth=1.5, linestyle='--', color='k', label='new bounds')]

    ax.legend(handles=legend_elements, loc='best')

    plt.text(0.14, 0.04, f'model: {model_name} - total time: {hours_conversion(dictionary["total_hours"])} - valid time: {hours_conversion(dictionary["hours_sleep"])} - new bounds time: {hours_conversion(dictionary["total_hours_new_bounds"])} - new bounds valid time: {hours_conversion(dictionary["hours_sleep_new_bounds"])} - signal quality in bounds: {dictionary["signal_quality_new_bounds"]} - threshold: {threshold} - valid: {dictionary["is_valid"]}', fontsize=11, transform=plt.gcf().transFigure)

    if plt_save_path != '':
        plt.savefig(f'{plt_save_path}/{model_name}/{threshold}/output_{title}.png', bbox_inches='tight')

    dictionary['plot'] = plt
    if view_graph:
        plt.show()
    plt.close(fig=fig)
        
    end = time.time()    # end timer variable used for the calculation of the total execution time 

    print('classification execution time =', round((end - start_class), 2), 'sec')
    print('total execution time =', round((end - start), 2), 'sec')

    enable_print()

    return dictionary


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default='', help='edf file path containing time series data')
    parser.add_argument('--model', type=str, default='LSTM', help='learning model architecture, either CNN, LSTM or KNN')
    parser.add_argument('--view_graph', dest='view_graph', action='store_true', help='invoke to view the output graph')
    parser.add_argument('--plt_save_path', type=str, default='', help='path to save a .png copy file of the output plot if desired')
    parser.set_defaults(view_graph=False)
    return parser.parse_args()


def main(p):
    out_dic = analysis_classification(**vars(p))
    print('-------- OUTPUT DICTIONARY --------')
    print(out_dic)
    print('-----------------------------------')


if __name__ == '__main__':
    # statement to avoid useless warnings during training
    # export TF_CPP_MIN_LOG_LEVEL=3
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    opt = parse_opt()
    main(p=opt)


    # main_dir = 'data/awake_sleep_analysis_edf_dfs'
    # out = sorted(os.listdir(main_dir))
    # for filename in out:

    #     mk3_file = f'{main_dir}/{filename}/{filename}_mk3.pkl'
    #     edf_file = f'{main_dir}/{filename}/{filename}_jawac.pkl'
    #     if os.path.exists(mk3_file) and os.path.exists(edf_file):

    #         df_mk3 = pd.read_pickle(mk3_file)
    #         df_jawac = pd.read_pickle(edf_file)
    #         df_jawac = df_jawac.resample('1S').median()

    #         # graph
    #         fig, ax = plt.subplots()
    #         fig.set_size_inches(18.5, 10.5)

    #         ax.plot(df_jawac.index.tolist(), df_jawac.data.tolist())
    #         ax.axhline(y=0, color='r', linewidth=1)
    #         ax.set(xlabel='time (s)', ylabel='opening (mm)', title=f'Jawac Signal - {filename}')
    #         ax.grid()

    #         df_jawac.reset_index(inplace=True)
    #         df_label = pd.DataFrame(columns=['start', 'end', 'label'])
    #         saved_label = df_jawac.iloc[0].label
    #         saved_start = df_jawac.iloc[0].times
    #         for idx, row in df_jawac.iterrows():
    #             if len(row) > 0:
    #                 if row.label != saved_label:
    #                     a_row = {'start': saved_start, 'end': row.times, 'label': saved_label}
    #                     df_label = df_label.append(a_row, ignore_index=True)
    #                     saved_label = row.label
    #                     saved_start = row.times
    #         a_row = {'start': saved_start, 'end': df_jawac.times.tolist()[-1], 'label': saved_label}
    #         df_label = df_label.append(a_row, ignore_index=True)
            
    #         for idx, row in df_mk3.iterrows():
    #             if row.label == 'OE':
    #                 plt.axvspan(row.start, row.end, facecolor='y', alpha=1)
    #             elif row.label == 'Out of Range':
    #                 plt.axvspan(row.start, row.end, facecolor='r', alpha=1)
    #         for idx, row in df_label.iterrows():
    #             if row.label == 0:
    #                 plt.axvspan(row.start, row.end, facecolor='b', alpha=0.3)
    #             elif row.label == 1:
    #                 plt.axvspan(row.start, row.end, facecolor='g', alpha=0.3)

    #         legend_elements = [Patch(facecolor='b', edgecolor='w', label='awake', alpha=0.3),
    #                             Patch(facecolor='g', edgecolor='w', label='sleep', alpha=0.3),
    #                             Patch(facecolor='y', edgecolor='w', label='oe', alpha=0.3),
    #                             Patch(facecolor='r', edgecolor='w', label='oor', alpha=0.3),]
    #         ax.legend(handles=legend_elements, loc='best')

    #         plt.show()
    #         # plt.savefig(f'/home/ckemdetry/Documents/Nomics/thesis_nomics/training/data/graphs/out_{filename}.png', bbox_inches='tight')
    #         # plt.close(fig=fig)
    #     else:
    #         print(edf_file)
    #         print(mk3_file)
    #         print('---')



    # main_dir = '/Users/clemdetry/Documents/Nomics'
    # for i in range(95):

    #     mk3_file = f'{main_dir}/marquages_finaux/{i}_complete.mk3'
    #     edf_file = f'{main_dir}/edf_clean/{i}_jawac.edf'
    #     if os.path.exists(mk3_file) and os.path.exists(edf_file):
    #         data_mk3 = open(mk3_file)
    #         lines = data_mk3.readlines()
    #         # start_record_time = lines[5].split(';')[0]
    #         # start_record_date = lines[5].split(';')[1]
    #         lines = lines[5:]
    #         col_names = ['start', 'end', 'label']
    #         df_mk3 = pd.DataFrame(columns = col_names)
    #         for line in lines:
    #             temp = pd.Series(extract_data_from_line(line), index = df_mk3.columns)
    #             df_mk3 = df_mk3.append(temp, ignore_index=True)

    #         # edf file
    #         block_print()
    #         raw_data = mne.io.read_raw_edf(edf_file)
    #         enable_print()
    #         data, times = raw_data[:]
    #         sec_interval = 1
    #         # times = string_datetime_conversion(times, start_record_time, start_record_date)
    #         times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])
    #         df_jawac = pd.DataFrame()
    #         df_jawac.insert(0, 'times', times)
    #         df_jawac.insert(1, 'data', data[0])
    #         df_jawac = df_jawac.set_index('times')
    #         df_jawac = df_jawac.resample(str(sec_interval)+'S').median()['data'].to_frame(name='data')

    #         # graph
    #         fig, ax = plt.subplots()
    #         fig.set_size_inches(18.5, 10.5)

    #         ax.plot(df_jawac.index.tolist(), df_jawac.data.tolist())
    #         ax.axhline(y=0, color='r', linewidth=1)
    #         ax.set(xlabel='time (s)', ylabel='opening (mm)', title=f'Jawac Signal - {edf_file}')
    #         ax.grid()

    #         for index, row in df_mk3.iterrows():
    #             if row['label'] == 'W':
    #                 plt.axvspan(row['start'], row['end'], facecolor='b', alpha=0.30)
    #             else:
    #                 plt.axvspan(row['start'], row['end'], facecolor='g', alpha=0.30)

    #         legend_elements = [Patch(facecolor='b', edgecolor='w', label='Awake', alpha=0.30),
    #                             Patch(facecolor='g', edgecolor='w', label='Sleep', alpha=0.30)]
    #         ax.legend(handles=legend_elements, loc='upper left')

    #         plt.show()
    #         # plt.savefig(f'/home/ckemdetry/Documents/Nomics/thesis_nomics/training/data/graphs/out_{filename}.png', bbox_inches='tight')
    #         # plt.close(fig=fig)
    #     else:
    #         print(edf_file)
    #         print(mk3_file)
    #         print('---')

    
    # import shutil
    # main_dir = '/Users/clemdetry/Documents/Nomics/Enregistrements_Fuites'
    # filenames = sorted(os.listdir(main_dir))
    # for f in filenames:
    #     filepath = f'{main_dir}/{f}/Hypnogram.txt'
    #     if os.path.exists(filepath):
    #         hypno = open(filepath)
    #         lines = hypno.readlines()
    #         if len(lines) > 1:
    #             print(filepath)
    #             if not os.path.exists(f'/Users/clemdetry/Documents/Nomics/thesis_nomics/data/awake_sleep_analysis/{f}'):
    #                 os.mkdir(f'/Users/clemdetry/Documents/Nomics/thesis_nomics/data/awake_sleep_analysis/{f}')
    #             mk3_file = open(f'/Users/clemdetry/Documents/Nomics/thesis_nomics/data/awake_sleep_analysis/{f}/{f}.mk3','w+')
    #             mk3_file.writelines(['MK_FILE_REV_0\n', 'ANJAW_REV=\n', 'IsReportUpToDate=No\n', 'Comment=\n', '-\n'])
    #             temp_time = lines[0].split(' ')[2][:-1] + ':000'
    #             edf_file = f'/Users/clemdetry/Documents/Nomics/edf_clean/{f}_jawac.edf'
    #             if os.path.exists(edf_file):
    #                 if not os.path.exists(f'/Users/clemdetry/Documents/Nomics/thesis_nomics/data/awake_sleep_analysis/{f}/{f}.edf'):
    #                     shutil.copy(edf_file, f'/Users/clemdetry/Documents/Nomics/thesis_nomics/data/awake_sleep_analysis/{f}/{f}.edf')
    #                 block_print()
    #                 raw_data = mne.io.read_raw_edf(edf_file)
    #                 enable_print()
    #                 data, times = raw_data[:]
    #                 times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])
    #                 start = str(times[0])
    #                 stop = str(times[-1])
    #                 out_line = []
    #                 out_line.append(start[11:19]+':000')
    #                 out_line.append(f'{start[8:10]}.{start[5:7]}.{start[2:4]}')
    #                 out_line.append(start[11:19]+':000')
    #                 out_line.append(f'{start[8:10]}.{start[5:7]}.{start[2:4]}')
    #                 out_line.append('START')
    #                 out = ';'.join(out_line)
    #                 mk3_file.write(out+'\n')
    #                 out_line = []
    #                 out_line.append(stop[11:19]+':000')
    #                 out_line.append(f'{stop[8:10]}.{stop[5:7]}.{stop[2:4]}')
    #                 out_line.append(stop[11:19]+':000')
    #                 out_line.append(f'{stop[8:10]}.{stop[5:7]}.{stop[2:4]}')
    #                 out_line.append('STOP')
    #                 out = ';'.join(out_line)
    #                 mk3_file.write(out+'\n')
    #                 date = str(raw_data.__dict__['info']['meas_date'])
    #                 temp_date = f'{date[8:10]}.{date[5:7]}.{date[2:4]}'
    #                 temp_label = lines[0].split(' ')[1]
    #                 for line in lines:
    #                     l = line.split(' ')
    #                     label = l[1]
    #                     t = l[2]
    #                     if label == temp_label:
    #                         continue
    #                     else:
    #                         if temp_label != 'W':
    #                             temp_label = label
    #                             if int((t[:-1]+':000')[:2]) < int(temp_time[:2]):
    #                                 num = int(temp_date[0:2])+1
    #                                 if num < 10:
    #                                     temp_date = f'0{num}{temp_date[2:]}'
    #                                 else:
    #                                     temp_date = f'{num}{temp_date[2:]}'
    #                             temp_time = t[:-1]+':000'
    #                             continue
    #                         out_line = []
    #                         out_line.append(temp_time)
    #                         out_line.append(temp_date)
    #                         if int((t[:-1]+':000')[:2]) < int(temp_time[:2]):
    #                             num = int(temp_date[0:2])+1
    #                             if num < 10:
    #                                 temp_date = f'0{num}{temp_date[2:]}'
    #                             else:
    #                                 temp_date = f'{num}{temp_date[2:]}'
    #                         temp_time = t[:-1]+':000'
    #                         out_line.append(temp_time)
    #                         out_line.append(temp_date)
    #                         out_line.append(temp_label)
    #                         temp_label = label
    #                         out = ';'.join(out_line)
    #                         mk3_file.write(out+'\n')
    #                 mk3_file.close()
    #                 print('-> done')


    # import shutil
    # for i in range(95):
    #     edf_file = f'/Users/clemdetry/Documents/Nomics/EDF_Simplifiés_Calibrés_Corrigés/{i}_jawac.edf'
    #     mk3_file = f'/Users/clemdetry/Documents/Nomics/EDF_Simplifiés_Calibrés_Corrigés/{i}_jawac.mk3'
    #     if os.path.exists(edf_file) and os.path.exists(mk3_file):
    #         if not os.path.exists(f'/Users/clemdetry/Documents/Nomics/thesis_nomics/data/awake_sleep_analysis/{i}'):
    #             os.mkdir(f'/Users/clemdetry/Documents/Nomics/thesis_nomics/data/awake_sleep_analysis/{i}')
    #         shutil.copy(edf_file, f'/Users/clemdetry/Documents/Nomics/thesis_nomics/data/awake_sleep_analysis/{i}/{i}.edf')
    #         shutil.copy(mk3_file, f'/Users/clemdetry/Documents/Nomics/thesis_nomics/data/awake_sleep_analysis/{i}/{i}.mk3')


    # Cmd test lines - patient data from 1 to 13
    # python3 code/classification/classify.py --edf 'code/classification/test_data/patient_data1.edf' --view_graph --model 'LSTM'
    # python3 code/classification/classify.py --edf 'code/classification/test_data/patient_data1.edf' --view_graph --model 'CNN'
