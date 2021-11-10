import os
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

from utils.util import datetime_conversion, f1_m, analysis_cutting, is_valid, block_print, enable_print, hours_conversion


def analysis_classification(edf, model, num_class):

    start = time.time()

    if num_class == 2:
        # time_split in minutes
        time_split = 2.0
        # time_resampling in seconds
        time_resampling = 1.0
        epochs = 5
        model_path = os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/models/{model.lower()}/binomial/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs/saved_model'
    else:
        # time_split in minutes
        time_split = 3.0
        # time_resampling in seconds
        time_resampling = 1.0
        epochs = 40
        model_path = os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/models/{model.lower()}/multinomial/split_{time_split}_resampling_{time_resampling}/{epochs}_epochs/saved_model'

    # edf file reading
    block_print()
    raw_data = mne.io.read_raw_edf(edf)
    enable_print()

    data, times = raw_data[:]
    times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])
    df_jawac = pd.DataFrame()
    df_jawac.insert(0, 'times', times)
    df_jawac.insert(1, 'data', data[0])
    df_jawac = df_jawac.resample(str(time_resampling) + 'S', on='times').median()['data'].to_frame(name='data')
    df_jawac.reset_index(inplace=True)

    start_class = time.time()

    # model loader
    if model.lower() in ['cnn', 'lstm']:
        if os.path.exists(model_path):
            model = load_model(model_path, compile=True, custom_objects={'f1_m': f1_m})
        else:
            raise Exception('model path does not exist')
    else:
        raise Exception('model should either be CNN or LSTM, found something else')

    size = int((60 / time_resampling) * time_split)
    X_test_seq = np.array([df_jawac.loc[i:i + size - 1, :].data for i in range(0, len(df_jawac), size)], dtype=object)

    X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, padding='post', dtype='float64')

    X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))

    df_jawac.set_index('times', inplace=True)

    predictions = model.predict(X_test_seq_pad)
    classes = []
    for item in predictions[:-1]:
        idx = np.argmax(item)
        if idx == 2:
            if item[idx] < 0.90:
                idx = 1
        classes.append((idx, item[idx]))
    # make sure the last window of the time series data is classified as invalid (sometimes issue with the padding)
    classes.append((0, 0))

    valid_total = 0
    invalid_total = 0
    for label in classes:
        if label[0] == 0:
            invalid_total += 1
        else:
            valid_total += 1
    valid_mean = valid_total / len(classes)
    invalid_mean = invalid_total / len(classes)

    print('--------')

    print('valid percentage:', round((valid_mean * 100), 2), '%')
    print('invalid percentage:', round((invalid_mean * 100), 2), '%')

    print('--------')

    print('valid hours:', hours_conversion((valid_total * time_split) / 60))
    print('invalid hours:', hours_conversion((invalid_total * time_split) / 60))

    print('--------')

    valid_hours, valid_rate, new_start, new_end = analysis_cutting(classes, df_jawac.index[0], df_jawac.index[-1], time_split, threshold=0.75)

    print('new start analysis time:', new_start)
    print('new end analysis time:', new_end)
    duration = (new_end - new_start)
    if new_start is not None and new_end is not None:
        print('analysis duration:', duration)
    else:
        print('analysis duration: Unknown')
    print('valid time in new bounds:', hours_conversion(valid_hours))
    print('valid rate in new bounds:', round(valid_rate*100, 2), '%')

    print('--------')

    print('is analysis valid:', is_valid(valid_hours, valid_rate))

    print('--------')

    # graph
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    # ax.plot(df_jawac.resample(str(graph_time_resampling)+'S').median()['data'].to_frame(name='data').index.tolist(), df_jawac.resample(str(graph_time_resampling)+'S').median()['data'].to_frame(name='data').data.tolist())
    ax.plot(df_jawac.index.tolist(), df_jawac.data.tolist())
    ax.axhline(y=0, color='r', linewidth=1)
    if new_start is not None and new_end is not None:
        ax.axvline(x=new_start, color='k', linewidth=2, linestyle='--')
        ax.axvline(x=new_end, color='k', linewidth=2, linestyle='--')
    title = ''
    for c in reversed(edf[:-4]):
        if c != '/':
            title += c
        else:
            break
    title = title[::-1]
    ax.set(xlabel='time', ylabel='opening (mm)', title=f'Jawac Signal - {title} - Valid hours in bounds = {hours_conversion(valid_hours)} - Valid: {is_valid(valid_hours, valid_rate)}')
    ax.grid()

    curr_time = raw_data.__dict__['info']['meas_date']
    for label in classes:
        if label[0] == 0:
            # plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='r', alpha=0.3*label[1])
            plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='r', alpha=0.25)
        elif label[0] == 1:
            # plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='g', alpha=0.3*label[1])
            plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='g', alpha=0.25)
        elif label[0] == 2:
            # plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='b', alpha=0.3*label[1])
            plt.axvspan(curr_time, curr_time + datetime.timedelta(minutes=time_split), facecolor='b', alpha=0.25)
        curr_time += datetime.timedelta(minutes=time_split)

    if num_class == 2:
        legend_elements = [Patch(facecolor='r', edgecolor='w', label='invalid area', alpha=0.2),
                           Patch(facecolor='g', edgecolor='w', label='valid area', alpha=0.2),
                           Line2D([0], [0], linewidth=1.5, linestyle='--', color='k', label='new bounds')]
    else:
        legend_elements = [Patch(facecolor='r', edgecolor='w', label='invalid area', alpha=0.2),
                           Patch(facecolor='g', edgecolor='w', label='valid area', alpha=0.2),
                           Patch(facecolor='b', edgecolor='w', label='awake area', alpha=0.2),
                           Line2D([0], [0], linewidth=1.5, linestyle='--', color='k', label='new bounds')]

    ax.legend(handles=legend_elements, loc='upper left')

    end = time.time()
    print('classification execution time =', round((end - start_class), 2), 'sec')
    print('total execution time =', round((end - start), 2), 'sec')
    print('--------')

    dictionary = {'percentage_signal_valid': round((valid_mean * 100), 2), 'percentage_signal_invalid': round((invalid_mean * 100), 2), 'hours_signal_valid': hours_conversion((valid_total * time_split) / 60), 'hours_signal_invalid': hours_conversion((invalid_total * time_split) / 60), 'new_bound_start': new_start, 'new_bound_end': new_end, 'duration_in_bounds': duration, 'hours_valid_in_bounds': hours_conversion(valid_hours), 'percentage_valid_in_bounds': round(valid_rate * 100, 2), 'is_valid': is_valid(valid_hours, valid_rate), 'plot': plt}

    # plt.savefig(os.path.dirname(os.path.abspath('classify_jawac.py')) + '/invalid_plt.png')
    # plt.show()

    return dictionary


def run(edf, model, num_class):
    out_dic = analysis_classification(edf=edf, model=model, num_class=num_class)
    print('is analysis valid:', out_dic['is_valid'])
    print('--------')
    print(out_dic)
    print('--------')
    out_dic['plot'].show()


def parse_opt():
    parser = argparse.ArgumentParser()
    # edf file path or object as input??
    parser.add_argument('--edf', type=str, default='', help='edf file path for time series extraction')
    parser.add_argument('--model', type=str, default='LSTM', help='deep learning model architecture - either CNN or LSTM')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes for classification, input 2 for (valid | invalid), 3 for (valid | invalid | awake)')
    return parser.parse_args()


def main(p):
    run(**vars(p))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # invalid
    # python classify_jawac.py --edf '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_invalid_analysis/2019_01_08_16_38_57_121-SER-14-369(R1)_FR_79y/2019_01_08_16_38_57_121-SER-14-369(R1)_FR_79y.edf' --model 'LSTM' --num_class 2
    # python classify_jawac.py --edf '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_invalid_analysis/2019_01_08_22_13_32_121-SER-15-407(R1)_FR_38y/2019_01_08_22_13_32_121-SER-15-407(R1)_FR_38y.edf' --model 'LSTM' --num_class 2
    # python classify_jawac.py --edf '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_invalid_analysis/2019_06_25_23_19_54_121-SER-17-575(R1)_FR_55y/2019_06_25_23_19_54_121-SER-17-575(R1)_FR_55y.edf' --model 'LSTM' --num_class 2
    # python classify_jawac.py --edf '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_invalid_analysis/2019_01_13_22_11_13_121-SER-14-346(R1)_FR_56y/2019_01_13_22_11_13_121-SER-14-346(R1)_FR_56y.edf' --model 'LSTM' --num_class 2
    # python classify_jawac.py --edf '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_invalid_analysis/2019_03_14_20_37_18_121-SER-15-420(R1)_FR_40y/2019_03_14_20_37_18_121-SER-15-420(R1)_FR_40y.edf' --model 'LSTM' --num_class 2
    # valid
    # python classify_jawac.py --edf '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_valid_analysis/2019_01_07_18_19_45_121-SER-13-271(R1)_FR_58y/2019_01_07_18_19_45_121-SER-13-271(R1)_FR_58y.edf' --model 'LSTM' --num_class 3
    # python classify_jawac.py --edf '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_valid_analysis/2019_01_31_23_56_20_121-SER-14-372(R2)_FR/2019_01_31_23_56_20_121-SER-14-372(R2)_FR.edf' --model 'LSTM' --num_class 3
    # python classify_jawac.py --edf '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_valid_analysis/2019_01_30_00_55_05_121-SER-16-495(R1)_FR_69y/2019_01_30_00_55_05_121-SER-16-495(R1)_FR_69y.edf' --model 'LSTM' --num_class 3
    # TO CHECK
    # python classify_jawac.py --edf '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_valid_analysis/2019_01_03_19_57_59_121-SER-16-463(R2)_NL/2019_01_03_19_57_59_121-SER-16-463(R2)_NL.edf' --model 'LSTM' --num_class 3
    # python classify_jawac.py --edf '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_valid_analysis/2019_01_07_15_53_00_121-SER-10-130(R3)_FR_36y/2019_01_07_15_53_00_121-SER-10-130(R3)_FR_36y.edf' --model 'LSTM' --num_class 3

    opt = parse_opt()
    main(p=opt)
