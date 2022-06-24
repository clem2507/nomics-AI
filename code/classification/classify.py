import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import mne
import time
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go

from keras.models import load_model
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')

from util import datetime_conversion, f1, analysis_cutting, is_valid, block_print, enable_print, hours_conversion, signal_quality, get_value_in_line


def analysis_classification(edf,
                            model_path_task1='models/task1/final/best/model-best.h5',
                            model_path_task2='models/task2/final/best/model-best.h5',
                            show_graph=False, 
                            en_print=False,
                            save=False):
    """
    Primary function for classifying a jawac patient signal

    Parameters:

    -edf (--edf): edf file path containing time series data
    -model_path_task1 (model_path_task1): corresponds to the path of the first task model weights
    -model_path_task2 (model_path_task2): corresponds to the paths of the second task model weights
    -show_graph (--show_graph): true to show the output graph, false to skip it and only use the output dictionary
    -en_print (--en_print): true to enable print in terminal
    -save (--save): true to save the output dictionary in a json file and the output dataframe in a csv file

    Returns:

    -dictionary: dictionary containing information computed on the jawac signal
        dictionary keys:    -'model_path_task1'
                            -'model_path_task2'
                            -'total_hours'
                            -'percentage_valid'
                            -'percentage_sleep'
                            -'percentage_awake'
                            -'percentage_invalid'
                            -'hours_valid'
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
    -df_jawac: corresponds to the pandas dataframe that contains the information on the predicted labels and probabilities
    """

    if not en_print:
        block_print()

    start = time.time()    # start timer variable used for the calculation of the total execution time 

    info_path_task1 = os.path.normpath(model_path_task1).split(os.sep)[:-2]
    info_path_task1.append('info.txt')
    info_path_task1 = '/'.join(info_path_task1)
    info_path_task1 = os.path.normpath(info_path_task1)
    model_path_task1 = os.path.normpath(model_path_task1)

    info_file_task1 = open(info_path_task1)
    lines = info_file_task1.readlines()
    lines = lines[3:9]
    
    segmentation_value = float(get_value_in_line(lines[0]))    # window segmentation value in second
    downsampling_value = float(get_value_in_line(lines[1]))    # signal downsampling value in second
    sliding_window = bool(int(get_value_in_line(lines[4])))    # boolean value for the sliding window state
    center_of_interest = int(get_value_in_line(lines[5]))    # center of interest size in seconds for the sliding window

    raw_data = mne.io.read_raw_edf(edf)    # edf file reading
    data, times = raw_data[:]    # edf file data extraction
    times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])    # conversion to usable date dtype 
    df_jawac = pd.DataFrame()    # dataframe creation to store time series data 
    df_jawac.insert(0, 'times', times)
    df_jawac.insert(1, 'data', data[0])
    df_jawac = df_jawac.resample(str(downsampling_value) + 'S', on='times').median()['data'].to_frame(name='data')
    df_jawac.reset_index(inplace=True)
    data = df_jawac.data.tolist()

    df_jawac['label'] = 1
    df_jawac['proba'] = 0.5

    start_class = time.time()    # start variable used for the calculation of the final classification time 

    print()
    print('Task 1...')
    print()

    timer_task1 = time.time()

    # this bloc of code divides the given time series into windows of 'size' number of data corresponding to the segmentation value in second
    size = int((1 / downsampling_value) * segmentation_value)
    if not sliding_window:
        X_test_seq = np.array([(df_jawac.loc[i:i + (size-1), :].data).to_numpy() for i in range(0, len(df_jawac), size)], dtype=object)
    else:
        size = int(segmentation_value)   # in sec
        X_test_seq = np.array([(df_jawac.loc[i:i + (size-1), :].data).to_numpy() for i in range(0, len(df_jawac), center_of_interest)], dtype=object)
    X_test_seq_temp = []
    for arr in X_test_seq:
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
    predictions = model.predict(X_test_seq_pad)    # model.predict classifies the X data by predicting the y labels
    # loop that runs through the list of model predictions to keep the highest predicted probability values
    for item in predictions[:-1]:
        idx = np.argmax(item)
        if idx == 0:
            if item[idx] > threshold:
                classes.append((idx, item[idx]))
            else:
                classes.append((1, 0.5))
        else:
            classes.append((idx, item[idx]))
    classes.append((0, 0.5))

    if sliding_window:
        mask = df_jawac.index[:(step_size//2)-(center_of_interest//2)]
        if classes[0][0] == 0:
            df_jawac.loc[mask, ['label']] = 0
        df_jawac.loc[mask, ['proba']] = classes[0][1]
    for i in range(len(classes)):
        if sliding_window:
            mask = df_jawac.index[(step_size//2)+(i*center_of_interest)-(center_of_interest//2):(step_size//2)+(i*center_of_interest)+(center_of_interest//2)]
        else:
            mask = df_jawac.index[i*step_size:i*step_size+step_size]
        if classes[i][0] == 0:
            df_jawac.loc[mask, ['label']] = 0
        df_jawac.loc[mask, ['proba']] = classes[i][1]

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

    print()
    print(f'---> Task 1 done in {round((time.time() - timer_task1), 2)} sec!')
    print()
    print('Task 2...')
    print()

    timer_task2 = time.time()

    info_path_task2 = os.path.normpath(model_path_task2).split(os.sep)[:-2]
    info_path_task2.append('info.txt')
    info_path_task2 = '/'.join(info_path_task2)
    info_path_task2 = os.path.normpath(info_path_task2)
    model_path_task2 = os.path.normpath(model_path_task2)

    info_file_task2 = open(info_path_task2)
    lines = info_file_task2.readlines()
    lines = lines[3:9]
    
    segmentation_value = float(get_value_in_line(lines[0]))    # window segmentation value in second
    downsampling_value = float(get_value_in_line(lines[1]))    # signal downsampling value in second
    sliding_window = bool(int(get_value_in_line(lines[4])))    # boolean value for the sliding window state
    center_of_interest = int(get_value_in_line(lines[5]))    # center of interest size in seconds for the sliding window

    df_jawac_only_valid = df_jawac[df_jawac['label']==1]
    data = df_jawac_only_valid.data.tolist()

    if len(data) > (segmentation_value*(1/downsampling_value)):

        # model loader
        if os.path.exists(model_path_task2):
            model = load_model(model_path_task2, compile=True, custom_objects={'f1': f1})
        else:
            raise Exception(model_path_task2, '-> model path does not exist')

        # this bloc of code divides the given time series into windows of 'size' number of data corresponding to the segmentation value in second
        size = int((1 / downsampling_value) * segmentation_value)
        if not sliding_window:
            X_test_seq = np.array([(data[i:i + size]) for i in range(0, len(data), size)], dtype=object)
        else:
            size = int(segmentation_value)   # in sec
            X_test_seq = np.array([(data[i:i + size]) for i in range(0, len(data), center_of_interest)], dtype=object)
        X_test_seq_temp = []
        for arr in X_test_seq:
            X_test_seq_temp.append(arr)
        X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq_temp, padding='post', dtype='float64')
        X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))

        threshold = 0.6
        classes = []
        step_size = int((1 / downsampling_value) * segmentation_value)
        predictions = model.predict(X_test_seq_pad)    # model.predict classifies the X data by predicting the y labels
        # loop that runs through the list of model predictions to keep the highest predicted probability values
        for item in predictions:
            idx = np.argmax(item)
            if idx == 0:
                if item[idx] > threshold:
                    classes.append((idx, item[idx]))
                else:
                    classes.append((1, 0.5))
            else:
                classes.append((idx, item[idx]))

        if sliding_window:
            mask = df_jawac_only_valid.index[:(step_size//2)-(center_of_interest//2)]
            if classes[0][0] == 0:
                df_jawac.loc[mask, ['label']] = 2
            df_jawac.loc[mask, ['proba']] = classes[0][1]
        for i in range(len(classes)):
            if sliding_window:
                mask = df_jawac_only_valid.index[(step_size//2)+(i*center_of_interest)-(center_of_interest//2):(step_size//2)+(i*center_of_interest)+(center_of_interest//2)]
            else:
                mask = df_jawac_only_valid.index[i*step_size:i*step_size+step_size]
            if classes[i][0] == 0:
                df_jawac.loc[mask, ['label']] = 2
            df_jawac.loc[mask, ['proba']] = classes[i][1]
        
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
        temp = pd.DataFrame(data = [[saved_start, df_jawac.times.iloc[-1], saved_label, np.mean(proba), len(proba)]], columns=['start', 'end', 'label', 'proba', 'data_num'])
        df_label = pd.concat([df_label, temp], ignore_index=True)

    # call of the function that proposes new bounds for the breakdown of the analysis for the diagnosis
    df_label, hours_sleep_in_bounds, sleep_rate_in_bound, new_start, new_end = analysis_cutting(df=df_label, 
                                                                             analysis_start=df_jawac.times.iloc[0], 
                                                                             analysis_end=df_jawac.times.iloc[-1], 
                                                                             downsampling_value=downsampling_value, 
                                                                             threshold=0.95)

    if new_start is not None and new_end is not None:
        duration = (new_end - new_start)
        print('new analysis duration:', duration)
    else:
        new_start = df_jawac.times.iloc[0]
        new_end = df_jawac.times.iloc[-1]
        duration = timedelta(seconds=0)
        print('analysis duration: Unknown')
    print('new start analysis time:', new_start)
    print('new end analysis time:', new_end)
    print('sleep time in new bounds:', hours_conversion(hours_sleep_in_bounds))
    print('sleep rate in new bounds:', round((sleep_rate_in_bound * 100), 2), '%')

    print('--------')
    
    print('total signal quality score (/100):', round((signal_quality(df_label) * 100), 2))

    print('--------')

    if 0 in df_jawac['label'].value_counts().keys():
        invalid_count = df_jawac['label'].value_counts()[0]
    else:
        invalid_count = 0
    if 1 in df_jawac['label'].value_counts().keys():
        sleep_count = df_jawac['label'].value_counts()[1]
    else:
        sleep_count = 0
    if 2 in df_jawac['label'].value_counts().keys():
        awake_count = df_jawac['label'].value_counts()[2]
    else:
        awake_count = 0

    print(f'invalid count: {invalid_count} --> {round((invalid_count/(invalid_count+sleep_count+awake_count)*100), 2)} %')
    print(f'sleep count: {sleep_count} --> {round((sleep_count/(invalid_count+sleep_count+awake_count)*100), 2)} %')
    print(f'awake count: {awake_count} --> {round((awake_count/(invalid_count+sleep_count+awake_count)*100), 2)} %')

    hours_invalid = round(((invalid_count * downsampling_value) / 3600), 2)
    hours_sleep = round(((sleep_count * downsampling_value) / 3600), 2)
    hours_awake = round(((awake_count * downsampling_value) / 3660), 2)

    print('--------')

    print('is analysis valid:', is_valid(hours_sleep, threshold=4))

    print('--------')

    total_invalid_rate = invalid_count/len(df_jawac)
    total_sleep_rate = sleep_count/len(df_jawac)
    total_awake_rate = awake_count/len(df_jawac)

    # creation of the output dictionary with computed information about the signal
    dictionary = {'model_path_task1': model_path_task1, 
                  'model_path_task2': model_path_task2, 
                  'total_hours': round(((times[-1] - times[0]).total_seconds() / 3600.0), 2), 
                  'percentage_valid': round(((total_sleep_rate + total_awake_rate) * 100), 2),
                  'percentage_sleep': round((total_sleep_rate * 100), 2), 
                  'percentage_awake': round((total_awake_rate * 100), 2), 
                  'percentage_invalid': round((total_invalid_rate * 100), 2), 
                  'hours_valid': hours_sleep+hours_awake, 
                  'hours_sleep': hours_sleep, 
                  'hours_awake': hours_awake, 
                  'hours_invalid': hours_invalid, 
                  'total_signal_quality': round((signal_quality(df_label) * 100), 2), 
                  'new_bound_start': str(new_start), 
                  'new_bound_end': str(new_end), 
                  'total_hours_new_bounds': round((duration.total_seconds() / 3600.0), 2), 
                  'hours_sleep_new_bounds': round(hours_sleep_in_bounds, 2), 
                  'percentage_sleep_new_bounds': round(sleep_rate_in_bound * 100, 2), 
                  'is_valid': is_valid(hours_sleep, threshold=4)}

    end = time.time()    # end timer variable used for the calculation of the total execution time 

    print('classification execution time =', round((end - start_class), 2), 'sec')
    print('total execution time =', round((end - start), 2), 'sec')

    if show_graph:
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

        fig.update_layout(title=go.layout.Title(
                                text=f'{title} <br><sup>Legend: {legend}</sup>',
                                xref='paper',
                                x=0))

        fig.layout.width = None
        fig.layout.height = None

        fig.show()

    enable_print()

    df_jawac.reset_index(inplace=True)

    if save:
        output_path_dic = os.path.normpath(edf)[:-3] + 'json'
        with open(output_path_dic, 'w') as file:
            file.write(json.dumps(dictionary))
        
        output_path_df = os.path.normpath(edf)[:-3] + 'csv'
        df_jawac.to_csv(output_path_df, 
                        header=['times', 'data', 'label', 'proba'], 
                        index=None, 
                        sep=';', 
                        mode='w')

    return dictionary, df_jawac


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default='', help='edf file path containing time series data')
    parser.add_argument('--model_path_task1', type=str, default='models/task1/final/best/model-best.h5', help='corresponds to the path of the first task model weights')
    parser.add_argument('--model_path_task2', type=str, default='models/task2/final/best/model-best.h5', help='corresponds to the paths of the second task model weights')
    parser.add_argument('--show_graph', dest='show_graph', action='store_true', help='invoke to show the output graph')
    parser.add_argument('--en_print', dest='en_print', action='store_true', help='invoke to enable print in terminal')
    parser.add_argument('--save', dest='save', action='store_true', help='invoke to save the output dictionary in a json file and the output dataframe in a csv file')
    parser.set_defaults(show_graph=False)
    parser.set_defaults(en_print=False)
    parser.set_defaults(save=False)
    return parser.parse_args()


def main(p):
    analysis_classification(**vars(p))


if __name__ == '__main__':
    # statement to avoid useless warnings during training
    # export TF_CPP_MIN_LOG_LEVEL=3
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Test cmd lines
    # python code/classification/classify.py --edf 'code/classification/test_data/patient_data3/patient_data3.edf' --show_graph

    opt = parse_opt()
    main(p=opt)
