import os
import sys
import time
import mne
import argparse
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/utils')

from util import datetime_conversion, f1_m, analysis_cutting, is_valid, block_print, enable_print, hours_conversion


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

    start = time.time()    # start timer variable used for the calculation of the total execution time 

    # if condition to distinct binary and binary classification
    if num_class == 2:
        segmentation_value = 1.0    # window segmentation value in minute
        downsampling_value = 3.0    # signal downsampling value in second
        model_path = os.path.dirname(os.path.abspath('util.py')) + f'/classification/test_models/{model.lower()}/binary/saved_model'
    else:
        segmentation_value = 1.0    # window segmentation value in minute
        downsampling_value = 1.0    # signal downsampling value in second
        model_path = os.path.dirname(os.path.abspath('util.py')) + f'/classification/test_models/{model.lower()}/multinomial/saved_model'

    block_print()
    raw_data = mne.io.read_raw_edf(edf)    # edf file reading
    enable_print()

    data, times = raw_data[:]    # edf file data extraction
    times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])    # conversion to usable date dtype 
    df_jawac = pd.DataFrame()    # dataframe creation to store time series data 
    df_jawac.insert(0, 'times', times)
    df_jawac.insert(1, 'data', data[0])
    df_jawac = df_jawac.resample(str(downsampling_value) + 'S', on='times').median()['data'].to_frame(name='data')
    df_jawac.reset_index(inplace=True)

    start_class = time.time()    # start variable used for the calculation of the final classification time 

    # model loader
    if model.lower() in ['cnn', 'lstm']:
        if os.path.exists(model_path):
            model = load_model(model_path, compile=True, custom_objects={'f1_m': f1_m})
        else:
            raise Exception('model path does not exist')
    else:
        raise Exception('model should either be CNN or LSTM, found something else')

    # this bloc of code divides the given time series into windows of 'size' number of data corresponding to the segmentation value in minute
    size = int((60 / downsampling_value) * segmentation_value)
    X_test_seq = np.array([df_jawac.loc[i:i + size - 1, :].data for i in range(0, len(df_jawac), size)], dtype=object)
    X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, padding='post', dtype='float64')
    X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))
    df_jawac.set_index('times', inplace=True)

    predictions = model.predict(X_test_seq_pad)    # model.predict classifies the X data by predicting the y labels
    classes = []    # classes list holds the predicted labels
    # loop that runs through the list of model predictions to keep the highest predicted probability values
    for item in predictions[:-1]:
        idx = np.argmax(item)
        classes.append((idx, item[idx]))
    classes.append((0, 0))

    valid_total = 0    # counter for the total number of valid regions found
    invalid_total = 0    # counter for the total number of invalid regions found
    for label in classes:
        if label[0] == 0:
            invalid_total += 1
        else:
            valid_total += 1
    valid_rate = valid_total / len(classes)
    invalid_rate = invalid_total / len(classes)

    print('--------')

    print('valid percentage:', round((valid_rate * 100), 2), '%')
    print('invalid percentage:', round((invalid_rate * 100), 2), '%')

    print('--------')

    print('valid hours:', hours_conversion((valid_total * segmentation_value) / 60))
    print('invalid hours:', hours_conversion((invalid_total * segmentation_value) / 60))

    print('--------')

    # call of the function that proposes new bounds for the breakdown of the analysis for the diagnosis
    valid_hours, valid_rate, new_start, new_end = analysis_cutting(classes, df_jawac.index[0], df_jawac.index[-1], segmentation_value, threshold=0.75)

    print('new start analysis time:', new_start)
    print('new end analysis time:', new_end)
    if new_start is not None and new_end is not None:
        duration = (new_end - new_start)
        print('new analysis duration:', duration)
    else:
        duration = 0
        print('analysis duration: Unknown')
    print('valid time in new bounds:', hours_conversion(valid_hours))
    print('valid rate in new bounds:', round(valid_rate * 100, 2), '%')

    print('--------')

    print('is analysis valid:', is_valid(valid_hours, valid_rate))

    print('--------')

    # creation of the output dictionary with computed information about the signal
    dictionary = {'percentage_signal_valid': round((valid_rate * 100), 2), 'percentage_signal_invalid': round((invalid_rate * 100), 2), 'hours_signal_valid': hours_conversion((valid_total * segmentation_value) / 60), 'hours_signal_invalid': hours_conversion((invalid_total * segmentation_value) / 60), 'new_bound_start': new_start, 'new_bound_end': new_end, 'duration_in_bounds': duration, 'hours_valid_in_bounds': hours_conversion(valid_hours), 'percentage_valid_in_bounds': round(valid_rate * 100, 2), 'is_valid': is_valid(valid_hours, valid_rate)}

    # if the graph is asked to be shown, this condition is entered
    if out_graph:
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
        ax.set(xlabel='time', ylabel='opening (mm)', title=f'Jawac Signal - {title} - Valid hours in bounds = {hours_conversion(valid_hours)} - Valid: {is_valid(valid_hours, valid_rate)}')
        ax.grid()

        curr_time = raw_data.__dict__['info']['meas_date']    # starting time of the analysis
        # graph background color based on signal classified label
        for label in classes:
            if label[0] == 0:
                plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=segmentation_value), facecolor='r', alpha=0.25)
            elif label[0] == 1:
                plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=segmentation_value), facecolor='g', alpha=0.25)
            elif label[0] == 2:
                plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=segmentation_value), facecolor='b', alpha=0.25)
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

        dictionary['plot'] = plt

        plt.show()

    end = time.time()    # end timer variable used for the calculation of the total execution time 

    print('classification execution time =', round((end - start_class), 2), 'sec')
    print('total execution time =', round((end - start), 2), 'sec')
    print('--------')
    print(dictionary)
    print('--------')

    return dictionary


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default='', help='edf file path containing time series data')
    parser.add_argument('--model', type=str, default='LSTM', help='deep learning model architecture, either CNN or LSTM')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes for classification, 2 for (valid | invalid), 3 for (valid | invalid | awake)')
    parser.add_argument('--out_graph', type=bool, default=False, help='true to show the output graph, false to skip it and only use the output dictionary')
    return parser.parse_args()


def main(p):
    analysis_classification(**vars(p))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    opt = parse_opt()
    main(p=opt)

    # Cmd test lines
    # python3 classification/classify_jawac.py --edf 'training/test_data/patient_data1.edf' --out_graph True --model 'LSTM' --num_class 3
    # python3 classification/classify_jawac.py --edf 'training/test_data/patient_data2.edf' --out_graph True --model 'LSTM' --num_class 3
    # python3 classification/classify_jawac.py --edf 'training/test_data/patient_data3.edf' --out_graph True --model 'LSTM' --num_class 3
