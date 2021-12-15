import os
import sys
import time
import mne
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import timedelta
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/utils')

from util import datetime_conversion, f1_m, analysis_cutting, is_valid, block_print, enable_print, hours_conversion, signal_quality, extract_data_from_line, string_datetime_conversion


def analysis_classification(edf, model, num_class, out_graph):
    """
    Primary function for classifying an edf patient file

    Parameters:

    -edf (--edf): edf file path containing time series data
    -model (--model): deep learning model architecture, either CNN or LSTM
    -num_class (--num_class): number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)
    -out_graph (--out_graph): true to show the output graph, false to skip it and only use the output dictionary

    Returns:

    -dictionary: dictionary containing information computed on the patient signal
        dictionary keys:    -'percentage_signal_valid'
                            -'percentage_signal_invalid'
                            -'hours_signal_valid'
                            -'hours_signal_invalid'
                            -'new_bound_start'
                            -'new_bound_end'
                            -'duration_in_bounds'
                            -'hours_valid_in_bounds'
                            -'percentage_valid_in_bounds'
                            -'is_valid'
    """

    block_print()

    start = time.time()    # start timer variable used for the calculation of the total execution time 

    # if condition to distinct binary and binary classification
    model_name = model.lower()
    saved_dir = os.path.dirname(os.path.abspath('util.py')) + f'/classification/models/{model_name}'
    most_recent_folder_path = sorted(Path(saved_dir).iterdir(), key=os.path.getmtime)[::-1]
    i = 0
    while True:
        model_path = str(most_recent_folder_path[i]) + '/saved_model'
        info_path = str(most_recent_folder_path[i]) + '/info.txt'
        info_file = open(info_path)
        lines = info_file.readlines()
        lines = lines[2:5]
        segmentation_value = float(lines[1][-6:-2])    # window segmentation value in minute
        downsampling_value = float(lines[2][-6:-2])    # signal downsampling value in second
        if int(lines[0][-3:-2]) == num_class:
            break
        i+=1
        if i == len(most_recent_folder_path):
            raise Exception(f'model path not found, please train a {num_class}-class {model_name} model')

    raw_data = mne.io.read_raw_edf(edf)    # edf file reading

    data, times = raw_data[:]    # edf file data extraction
    times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])    # conversion to usable date dtype 
    df_jawac = pd.DataFrame()    # dataframe creation to store time series data 
    df_jawac.insert(0, 'times', times)
    df_jawac.insert(1, 'data', data[0])
    df_jawac = df_jawac.resample(str(downsampling_value) + 'S', on='times').median()['data'].to_frame(name='data')
    df_jawac.reset_index(inplace=True)

    start_class = time.time()    # start variable used for the calculation of the final classification time 

    # this bloc of code divides the given time series into windows of 'size' number of data corresponding to the segmentation value in minute
    size = int((60 / downsampling_value) * segmentation_value)
    X_test_seq = np.array([(df_jawac.loc[i:i + size - 1, :].data).to_numpy() for i in range(0, len(df_jawac), size)], dtype=object)
    X_test_seq_temp = []
    for arr in X_test_seq:
        # arr = arr.append(pd.Series([np.var(arr)*1000]), ignore_index=True)
        arr = np.append(arr, pd.Series([np.var(arr)*1000]))
        X_test_seq_temp.append(arr)
        # X_test_seq_temp.append(pd.Series([np.var(arr), np.median(arr)]))
    # X_test_seq_temp = np.array(X_test_seq_temp, dtype=list)
    X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq_temp, padding='post', dtype='float64')
    X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))
    df_jawac.set_index('times', inplace=True)

    # model loader
    if os.path.exists(model_path):
        if model_name in ['cnn', 'lstm']:
            model = load_model(model_path, compile=True, custom_objects={'f1_m': f1_m})
        else:
            X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1]))
            model = pickle.load(open(model_path, 'rb'))
    else:
        raise Exception('model path does not exist')

    predictions = model.predict(X_test_seq_pad)    # model.predict classifies the X data by predicting the y labels
    classes = []    # classes list holds the predicted labels
    # loop that runs through the list of model predictions to keep the highest predicted probability values
    for item in predictions[:-1]:
        idx = np.argmax(item)
        if model_name in ['cnn', 'lstm']:
            if idx == 0:
                if item[idx] > 0.9:
                    print(item[idx])
                    classes.append((idx, item[idx]))
                else:
                    classes.append((1, 0.5))
            else:
                classes.append((idx, item[idx]))
        else:
            classes.append((idx, 1))
    classes.append((0, 0))

    valid_total = 0    # counter for the total number of valid regions found
    invalid_total = 0    # counter for the total number of invalid regions found
    for label in classes:
        if label[0] == 0:
            invalid_total += 1
        else:
            valid_total += 1
    total_valid_rate = valid_total / len(classes)
    total_invalid_rate = invalid_total / len(classes)

    print('--------')

    print('valid percentage:', round((total_valid_rate * 100), 2), '%')
    print('invalid percentage:', round((total_invalid_rate * 100), 2), '%')

    print('--------')

    print('valid hours:', hours_conversion((valid_total * segmentation_value) / 60))
    print('invalid hours:', hours_conversion((invalid_total * segmentation_value) / 60))

    print('--------')
    
    print('total signal quality', round((signal_quality(classes) * 100), 2))

    print('--------')

    # call of the function that proposes new bounds for the breakdown of the analysis for the diagnosis
    new_bounds_classes, valid_hours, valid_rate, new_start, new_end = analysis_cutting(classes, df_jawac.index[0], df_jawac.index[-1], segmentation_value, threshold=0.95)

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

    # creation of the output dictionary with computed information about the signal
    dictionary = {'model_path': model_path, 'total_hours': round(((times[-1] - times[0]).total_seconds() / 3600.0), 2), 'percentage_valid': round((total_valid_rate * 100), 2), 'percentage_invalid': round((total_invalid_rate * 100), 2), 'hours_valid': round(((valid_total * segmentation_value) / 60), 2), 'hours_invalid': round(((invalid_total * segmentation_value) / 60), 2), 'total_signal_quality': round((signal_quality(classes) * 100), 2), 'new_bound_start': new_start, 'new_bound_end': new_end, 'total_hours_new_bounds': round((duration.total_seconds() / 3600.0), 2), 'hours_valid_new_bounds': round(valid_hours, 2), 'percentage_valid_new_bounds': round(valid_rate * 100, 2), 'signal_quality_new_bounds': round((signal_quality(new_bounds_classes) * 100), 2), 'is_valid': is_valid(times[-1] - times[0], valid_hours)}

    # graph
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

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

    curr_time = raw_data.__dict__['info']['meas_date']    # starting time of the analysis
    # graph background color based on signal classified label
    for label in classes:
        if label[0] == 0:
            plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=segmentation_value), facecolor='r', alpha=0.3*label[1])
            # plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=segmentation_value), facecolor='r', alpha=0.25)
        elif label[0] == 1:
            plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=segmentation_value), facecolor='g', alpha=0.3*label[1])
            # plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=segmentation_value), facecolor='g', alpha=0.25)
        elif label[0] == 2:
            plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=segmentation_value), facecolor='b', alpha=0.3*label[1])
            # plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=segmentation_value), facecolor='b', alpha=0.25)
        curr_time += datetime.timedelta(minutes=segmentation_value)

    # legend
    if num_class == 2:
        legend_elements = [Patch(facecolor='r', edgecolor='w', label='invalid area', alpha=0.2),
                            Patch(facecolor='g', edgecolor='w', label='valid area', alpha=0.2),
                            Line2D([0], [0], linewidth=1.5, linestyle='--', color='k', label='new bounds')]
    else:
        legend_elements = [Patch(facecolor='r', edgecolor='w', label='invalid area', alpha=0.2),
                            Patch(facecolor='g', edgecolor='w', label='valid area', alpha=0.2),
                            Patch(facecolor='b', edgecolor='w', label='awake area', alpha=0.2),
                            Line2D([0], [0], linewidth=1.5, linestyle='--', color='k', label='new bounds')]

    ax.legend(handles=legend_elements, loc='best')

    plt.text(0.23, 0.04, f'total time: {hours_conversion(dictionary["total_hours"])} - valid time: {hours_conversion(dictionary["hours_valid"])} - new bounds time: {hours_conversion(dictionary["total_hours_new_bounds"])} - new bounds valid time: {hours_conversion(dictionary["hours_valid_new_bounds"])} - valid: {dictionary["is_valid"]}', fontsize=12, transform=plt.gcf().transFigure)

    plt.savefig(f'/home/ckemdetry/Documents/Nomics/thesis_nomics/training/data/output_graphs/output_{title}.png', bbox_inches='tight')

    dictionary['plot'] = plt
    if out_graph:
        plt.show()
    plt.close(fig=fig)
        

    end = time.time()    # end timer variable used for the calculation of the total execution time 

    print('classification execution time =', round((end - start_class), 2), 'sec')
    print('total execution time =', round((end - start), 2), 'sec')

    return dictionary


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default='', help='edf file path containing time series data')
    parser.add_argument('--model', type=str, default='LSTM', help='deep learning model architecture, either CNN or LSTM')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)')
    parser.add_argument('--view_graph', dest='out_graph', action='store_true', help='invoke to view the output graph')
    parser.set_defaults(out_graph=False)
    return parser.parse_args()


def main(p):
    out_dic = analysis_classification(**vars(p))
    enable_print()
    print('------- OUTPUT DICTIONARY -------')
    print(out_dic)
    print('---------------------------------')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    opt = parse_opt()
    main(p=opt)

    # directory = '/home/ckemdetry/Documents/Nomics/thesis_nomics/training/data/analysis'
    # filenames = sorted(os.listdir(directory))
    # for f in filenames:
    #     if not f.startswith('.'):
    #         edf_path = f'{directory}/{f}/{f}.edf'
    #         if os.path.exists(edf_path):
    #             analysis_classification(edf_path, 'lstm', 2, False)

    # Cmd test lines
    # python3 classification/classify_jawac.py --edf 'classification/test_data/patient_data1.edf' --view_graph --model 'LSTM'
    # python3 classification/classify_jawac.py --edf 'classification/test_data/patient_data2.edf' --view_graph --model 'LSTM'
    # python3 classification/classify_jawac.py --edf 'classification/test_data/patient_data3.edf' --view_graph --model 'LSTM'
    # python3 classification/classify_jawac.py --edf 'classification/test_data/patient_data4.edf' --view_graph --model 'LSTM'
    # python3 classification/classify_jawac.py --edf 'classification/test_data/patient_data5.edf' --view_graph --model 'LSTM'
    # python3 classification/classify_jawac.py --edf 'classification/test_data/patient_data6.edf' --view_graph --model 'LSTM'
    # python3 classification/classify_jawac.py --edf 'classification/test_data/patient_data7.edf' --view_graph --model 'LSTM'
