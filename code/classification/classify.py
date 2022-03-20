import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import mne
import time
import math
import pickle
import argparse
import datetime
import mplcursors
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
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
                            -'is_valid'
                            -'plot'
    """

    # block_print()

    start = time.time()    # start timer variable used for the calculation of the total execution time 

    model_name = model.lower()
    saved_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task1/{model_name}'

    most_recent_folder_path = sorted(Path(saved_dir).iterdir(), key=os.path.getmtime)[::-1]
    most_recent_folder_path = [name for name in most_recent_folder_path if not (str(name).split('/')[-1]).startswith('.')]

    model_path = str(most_recent_folder_path[0]) + '/best'
    info_path = str(most_recent_folder_path[0]) + '/info.txt'
    info_file = open(info_path)
    lines = info_file.readlines()
    lines = lines[3:11]
    
    segmentation_value = float(get_value_in_line(lines[0]))    # window segmentation value in minute
    downsampling_value = float(get_value_in_line(lines[1]))    # signal downsampling value in second
    batch_size = int(get_value_in_line(lines[3]))    # batch size value for prediction
    standard_scale = bool(int(get_value_in_line(lines[4])))    # boolean value for the standardizatin of the data state
    stateful = bool(int(get_value_in_line(lines[5])))    # boolean value for the stateful model state
    sliding_window = bool(int(get_value_in_line(lines[6])))    # boolean value for the sliding window state
    center_of_interest = int(get_value_in_line(lines[7]))    # center of interest size in seconds for the sliding window

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

    print('Task 1...')

    # this bloc of code divides the given time series into windows of 'size' number of data corresponding to the segmentation value in minute
    if not stateful:
        size = int((60 / downsampling_value) * segmentation_value)
        if not sliding_window:
            X_test_seq = np.array([(df_jawac.loc[i:i + (size-1), :].data).to_numpy() for i in range(0, len(df_jawac), size)], dtype=object)
        else:
            size = int(segmentation_value * 60)   # in sec
            X_test_seq = np.array([(df_jawac.loc[i:i + (size-1), :].data).to_numpy() for i in range(0, len(df_jawac), center_of_interest)], dtype=object)
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
        step_size = 60
        minutes_per_class = 1
        for i in range(0, len(data), batch_size*step_size):
            if i+(batch_size*step_size) < len(data):
                y_pred, *r = model.predict_on_batch(np.reshape(data[i:i+batch_size*step_size], (batch_size, step_size, 1)))
                for label in y_pred:
                    pred_label = round(label[0])
                    if pred_label == 0:
                        if 1-y_pred[0][0] > threshold:
                            classes.append((pred_label, 1-label[0]))
                        else:
                            classes.append((1, 0.5))
                    else:
                        classes.append((label, label[0]))
    classes.append((0, 0.5))

    step_size = int((60 / downsampling_value) * segmentation_value)
    df_jawac['label'] = [1 for n in range(len(df_jawac))]
    df_jawac['proba'] = [0.5 for n in range(len(df_jawac))]
    for i in range(len(classes)):
        if classes[i][0] == 0:
            df_jawac.loc[(i*step_size):(i*step_size)+step_size, 'label'] = 0
            df_jawac.loc[(i*step_size):(i*step_size)+step_size, 'proba'] = classes[i][1]

    min_gap_time = int((60 / downsampling_value) * 2)   # in minutes
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

    valid_total = df_jawac['label'].value_counts()[1]    # counter for the total number of valid regions found
    invalid_total = df_jawac['label'].value_counts()[0]    # counter for the total number of invalid regions found

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

    if not is_valid(times[-1] - times[0], (valid_total * minutes_per_class) / 60):
        raise Exception('---> Analysis invalid')

    print('---> Task 1 done!')
    print('Task 2...')
    print()

    saved_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task2/{model_name}'

    most_recent_folder_path = sorted(Path(saved_dir).iterdir(), key=os.path.getmtime)[::-1]
    most_recent_folder_path = [name for name in most_recent_folder_path if not (str(name).split('/')[-1]).startswith('.')]

    model_path = str(most_recent_folder_path[0]) + '/best'
    info_path = str(most_recent_folder_path[0]) + '/info.txt'
    info_file = open(info_path)
    lines = info_file.readlines()
    lines = lines[3:11]
    
    segmentation_value = float(get_value_in_line(lines[0]))    # window segmentation value in minute
    downsampling_value = float(get_value_in_line(lines[1]))    # signal downsampling value in second
    batch_size = int(get_value_in_line(lines[3]))    # batch size value for prediction
    standard_scale = bool(int(get_value_in_line(lines[4])))    # boolean value for the standardizatin of the data state
    stateful = bool(int(get_value_in_line(lines[5])))    # boolean value for the stateful model state
    sliding_window = bool(int(get_value_in_line(lines[6])))    # boolean value for the sliding window state
    center_of_interest = int(get_value_in_line(lines[7]))    # center of interest size in seconds for the sliding window

    # model loader
    if os.path.exists(model_path):
        if model_name in ['cnn', 'lstm']:
            model = load_model(model_path, compile=True, custom_objects={'f1_m': f1_m})
    else:
        raise Exception('model path does not exist')

    X = df_jawac[df_jawac['label']==1].data.tolist()

    # this bloc of code divides the given time series into windows of 'size' number of data corresponding to the segmentation value in minute
    if not stateful:
        size = int((60 / downsampling_value) * segmentation_value)
        if not sliding_window:
            X_test_seq = np.array([(X[i:i + size]) for i in range(0, len(X), size)], dtype=object)
        else:
            size = int(segmentation_value * 60)   # in sec
            X_test_seq = np.array([(X[i:i + size]) for i in range(0, len(X), center_of_interest)], dtype=object)
        X_test_seq_temp = []
        for arr in X_test_seq:
            # arr = np.append(arr, pd.Series([np.var(arr)*1000]))
            X_test_seq_temp.append(arr)
        X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq_temp, padding='post', dtype='float64')
        X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))

    threshold = 0.9
    minutes_per_class = (batch_size * downsampling_value)/60
    classes = []
    step_size = int((60 / downsampling_value) * segmentation_value)
    if not stateful:
        minutes_per_class = segmentation_value
        predictions = model.predict(X_test_seq_pad)    # model.predict classifies the X data by predicting the y labels
        # loop that runs through the list of model predictions to keep the highest predicted probability values
        for item in predictions[:-1]:
            idx = np.argmax(item)
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
        step_size = 60
        for i in range(0, len(X), batch_size*step_size):
            if i+(batch_size*step_size) < len(X):
                y_pred, *r = model.predict_on_batch(np.reshape(X[i:i+batch_size*step_size], (batch_size, step_size, 1)))
                for label in y_pred:
                    pred_label = round(label[0])
                    # if pred_label == 1:
                    #     if y_pred[0][0] > threshold:
                    #         classes.append((pred_label, label[0]))
                    #     else:
                    #         classes.append((0, 0.5))
                    # else:
                    #     classes.append((pred_label, 1-label[0]))
                    if pred_label == 0:
                        if 1-y_pred[0][0] > threshold:
                            classes.append((pred_label, 1-label[0]))
                        else:
                            classes.append((1, 0.5))
                    else:
                        classes.append((label, label[0]))

    df_jawac_only_valid = df_jawac[df_jawac['label']==1]
    for i in range(len(classes)):
        if sliding_window:
            mask = df_jawac_only_valid.index[(step_size//2)+(i*center_of_interest)-(center_of_interest//2):(step_size//2)+(i*center_of_interest)+(center_of_interest//2)]
        else:
            mask = df_jawac_only_valid.index[i*step_size:i*step_size+step_size]
        if classes[i][0] == 0:
            df_jawac.loc[mask, ['label']] = 2
        df_jawac.loc[mask, ['proba']] = classes[i][1]

    min_gap_time = int((60 / downsampling_value) * 2)   # in minutes
    previous_label = df_jawac.label[0]
    idx = []
    for i in range(len(df_jawac.label)):
        idx.append(i)
        if df_jawac.label[i] != previous_label:
            if len(idx) <= min_gap_time:
                if previous_label == 1:
                    df_jawac.loc[idx, 'label'] = 2
                    df_jawac.loc[idx, 'proba'] = 0.5
                elif previous_label == 2:
                    df_jawac.loc[idx, 'label'] = 1
                    df_jawac.loc[idx, 'proba'] = 0.5
            else:
                idx = []
                idx.append(i)
        previous_label = df_jawac.label[i]

    df_label = pd.DataFrame(columns=['start', 'end', 'label', 'proba', 'data_num'])
    saved_label = df_jawac.iloc[0].label
    saved_start = df_jawac.iloc[0].times
    proba = []
    for idx, row in df_jawac.iterrows():
        if len(row) > 0:
            proba.append(row.proba)
            if row.label != saved_label:
                a_row = {'start': saved_start, 'end': row.times, 'label': saved_label, 'proba': np.mean(proba), 'data_num': len(proba)}
                df_label = df_label.append(a_row, ignore_index=True)
                saved_label = row.label
                saved_start = row.times
                proba = []
    a_row = {'start': saved_start, 'end': df_jawac.times.tolist()[-1], 'label': saved_label, 'proba': np.mean(proba), 'data_num': len(proba)}
    df_label = df_label.append(a_row, ignore_index=True)

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

    total_invalid_rate = invalid_count/len(df_jawac)
    total_valid_rate = valid_count/len(df_jawac)
    total_awake_rate = awake_count/len(df_jawac)

    # creation of the output dictionary with computed information about the signal
    dictionary = {'model_path': model_path, 'total_hours': round(((times[-1] - times[0]).total_seconds() / 3600.0), 2), 'percentage_sleep': round((total_valid_rate * 100), 2), 'percentage_awake': round((total_awake_rate * 100), 2), 'percentage_invalid': round((total_invalid_rate * 100), 2), 'hours_sleep': round(((valid_count * downsampling_value) / 3600), 2), 'hours_awake': round(((awake_count * downsampling_value) / 3660), 2), 'hours_invalid': round(((invalid_count * downsampling_value) / 3600), 2), 'total_signal_quality': round((signal_quality(df_label) * 100), 2), 'new_bound_start': new_start, 'new_bound_end': new_end, 'total_hours_new_bounds': round((duration.total_seconds() / 3600.0), 2), 'hours_sleep_new_bounds': round(sleep_hours, 2), 'percentage_sleep_new_bounds': round(sleep_rate * 100, 2), 'is_valid': is_valid(times[-1] - times[0], sleep_hours)}

    end = time.time()    # end timer variable used for the calculation of the total execution time 

    print('classification execution time =', round((end - start_class), 2), 'sec')
    print('total execution time =', round((end - start), 2), 'sec')

    # graph
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    df_jawac.set_index('times', inplace=True)

    # plot of the time series values
    ax.plot(df_jawac.index.tolist(), df_jawac.data.tolist())
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

    curr_time = raw_data.__dict__['info']['meas_date']    # starting time of the analysis -- Not needed anymore
    # graph background color based on signal classified label
    for idx, row in df_label.iterrows():
        if row.label == 0:
            plt.axvspan(row.start, row.end, facecolor='r', alpha=0.3*row.proba)
        elif row.label == 1:
            plt.axvspan(row.start, row.end, facecolor='g', alpha=0.3*row.proba)
        elif row.label == 2:
            plt.axvspan(row.start, row.end, facecolor='b', alpha=0.3*row.proba)

    # legend
    legend_elements = [Patch(facecolor='r', edgecolor='w', label='invalid', alpha=0.25),
                    Patch(facecolor='g', edgecolor='w', label='sleep', alpha=0.25),
                    Patch(facecolor='b', edgecolor='w', label='awake', alpha=0.25),
                    Line2D([0], [0], linewidth=1.5, linestyle='--', color='k', label='new bounds')]

    ax.legend(handles=legend_elements, loc='best')

    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(round(df_jawac.proba[int(sel.index)], 4)))

    plt.text(0.14, 0.04, f'model: {model_name} - total time: {hours_conversion(dictionary["total_hours"])} - hours sleep: {hours_conversion(dictionary["hours_sleep"])} - new bounds time: {hours_conversion(dictionary["total_hours_new_bounds"])} - new bounds sleep time: {hours_conversion(dictionary["hours_sleep_new_bounds"])} - total signal quality: {dictionary["total_signal_quality"]} - valid: {dictionary["is_valid"]}', fontsize=11, transform=plt.gcf().transFigure)

    if plt_save_path != '':
        plt.savefig(f'{plt_save_path}/{model_name}/{threshold}/output_{title}.png', bbox_inches='tight')

    dictionary['plot'] = plt
    if view_graph:
        plt.show()
    plt.close(fig=fig)

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


    # Test cmd lines - patient data from 1 to 13
    # python3 code/classification/classify.py --view_graph --model 'LSTM' --edf 'code/classification/test_data/patient_data1.edf'
    # python3 code/classification/classify.py --view_graph --model 'CNN' --edf 'code/classification/test_data/patient_data1.edf'


    opt = parse_opt()
    main(p=opt)
