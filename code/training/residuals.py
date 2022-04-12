import os
import sys
import argparse
import mplcursors
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from keras.models import load_model
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')

from util import f1, get_value_in_line, enable_print


def find_residuals(analysis_directory, model_directory, model_name, task):
    """
    Method used to find the residuals between the model predictions and the actual labels

    Parameters:

    -analysis_directory: directory path with analysis to use for testing neural network and finding residuals
    -model_directory: directory path with model to use
    -model_name: learning model architecture - either CNN or LSTM
    -task: corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep
    """

    # model loader
    if model_directory == '':
        saved_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task{task}/{model_name.lower()}'
        most_recent_folder_path = sorted(Path(saved_dir).iterdir(), key=os.path.getmtime)[::-1]
        most_recent_folder_path = [name for name in most_recent_folder_path if not (str(name).split('/')[-1]).startswith('.')]
        model_path = str(most_recent_folder_path[0]) + '/best/model-best.h5'
        info_path = str(most_recent_folder_path[0]) + '/info.txt'
    else:
        model_path = f'{model_directory}/best/model-best.h5'
        info_path = f'{model_directory}/info.txt'
    if os.path.exists(model_path):
        model = load_model(model_path, compile=True, custom_objects={'f1': f1})
    else:
        raise Exception(model_path, '-> model path does not exist')

    if os.path.exists(info_path):
        info_file = open(info_path)
        lines = info_file.readlines()
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
    else:
        raise Exception(model_path, '-> info file path does not exist')

    # analysis loader
    analysis_directory = f'{analysis_directory}_edf_dfs'
    if not os.path.exists(model_path):
        raise Exception(model_path, '-> analysis directory path does not exist')
    for filename in sorted(os.listdir(analysis_directory)):
        mk3_file = f'{analysis_directory}/{filename}/{filename}_mk3.pkl'
        edf_file = f'{analysis_directory}/{filename}/{filename}_jawac.pkl'
        if os.path.exists(mk3_file) and os.path.exists(edf_file):
            df_mk3 = pd.read_pickle(mk3_file)
            df_jawac = pd.read_pickle(edf_file)
            df_jawac = df_jawac.resample(f'{downsampling_value}S').median()
            df_jawac['label'] = [1 for n in range(len(df_jawac))]
            df_jawac['pred_label'] = [1 for n in range(len(df_jawac))]
            df_jawac.index = pd.to_datetime(df_jawac.index).tz_localize(None)
            for idx, row in df_mk3.iterrows():
                mask = (df_jawac.index >= row.start) & (df_jawac.index <= row.end)
                if task == 1:
                    df_jawac.loc[mask, ['label']] = 0
                else:
                    if row.label == 'W':
                        df_jawac.loc[mask, ['label']] = 0
                    else:
                        df_jawac = df_jawac[~mask]

            df_jawac.reset_index(inplace=True)

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
                    if not return_sequences:
                        arr = np.append(arr, pd.Series([np.var(arr)*1000]))
                    X_test_seq_temp.append(arr)
                X_test_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq_temp, padding='post', dtype='float64')
                X_test_seq_pad = np.reshape(X_test_seq_pad, (X_test_seq_pad.shape[0], X_test_seq_pad.shape[1], 1))
            
            classes = []    # classes list holds the predicted labels
            threshold = 0.5    # above this threshold, the model invalid prediction are kept, otherwise considered as valid
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
                step_size = int(((1 / downsampling_value) * segmentation_value) * batch_size)
                data = df_jawac.data.tolist()
                if not full_sequence:
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
                else:
                    y_pred, *r = model.predict_on_batch(np.reshape(data, (batch_size, -1, 1)))
                    for label in y_pred:
                        pred_label = round(label[0])
                        if pred_label == 0:
                            if 1-y_pred[0][0] > threshold:
                                classes.append((pred_label, 1-label[0]))
                            else:
                                classes.append((1, 0.5))
                        else:
                            classes.append((label, label[0]))

            df_jawac['proba'] = [1 for n in range(len(df_jawac))]
            df_jawac['pred_proba'] = [0.5 for n in range(len(df_jawac))]

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
                    if task == 1:
                        df_jawac.loc[mask, ['pred_label']] = 0
                    else:
                        df_jawac.loc[mask, ['pred_label']] = 2
                df_jawac.loc[mask, ['pred_proba']] = classes[i][1]

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

            df_pred_label = pd.DataFrame(columns=['start', 'end', 'label', 'proba', 'data_num'])
            saved_label = df_jawac.iloc[0].pred_label
            saved_start = df_jawac.iloc[0].times
            proba = []
            for idx, row in df_jawac.iterrows():
                if len(row) > 0:
                    proba.append(row.pred_proba)
                    if row.label != saved_label:
                        temp = pd.DataFrame(data = [[saved_start, row.times, saved_label, np.mean(proba), len(proba)]], columns=['start', 'end', 'label', 'proba', 'data_num'])
                        df_pred_label = pd.concat([df_pred_label, temp], ignore_index=True)
                        saved_label = row.pred_label
                        saved_start = row.times
                        proba = []
            if len(proba) > 0:
                temp = pd.DataFrame(data = [[saved_start, df_jawac.times.tolist()[-1], saved_label, np.mean(proba), len(proba)]], columns=['start', 'end', 'label', 'proba', 'data_num'])
                df_pred_label = pd.concat([df_pred_label, temp], ignore_index=True)

            df_jawac.set_index('times', inplace=True)
            fig, ax = plt.subplots(nrows=2, ncols=1)
            fig.set_size_inches(18.5, 12.5)
            count = 0
            for df in [df_label, df_pred_label]:

                ax[count].plot(df_jawac.index.tolist(), df_jawac.data.tolist())
                ax[count].axhline(y=0, color='r', linewidth=1)
                if count == 0:
                    ax[count].set(xlabel='time (s)', ylabel='opening (mm)', title=f'Jawac Signal - {filename} - actual')
                else:
                    ax[count].set(xlabel='time (s)', ylabel='opening (mm)', title=f'Jawac Signal - {filename} - predicted')
                ax[count].grid()

                # graph background color based on signal classified label
                for idx, row in df.iterrows():
                    if row.label == 0:
                        ax[count].axvspan(row.start, row.end, facecolor='r', alpha=0.3*row.proba)
                    elif row.label == 1:
                        ax[count].axvspan(row.start, row.end, facecolor='g', alpha=0.3*row.proba)
                    elif row.label == 2:
                        ax[count].axvspan(row.start, row.end, facecolor='b', alpha=0.3*row.proba)

                # legend
                legend_elements = [Patch(facecolor='r', edgecolor='w', label='invalid', alpha=0.25),
                                Patch(facecolor='g', edgecolor='w', label='sleep', alpha=0.25),
                                Patch(facecolor='b', edgecolor='w', label='awake', alpha=0.25),
                                Line2D([0], [0], linewidth=1.5, linestyle='--', color='k', label='new bounds')]

                ax[count].legend(handles=legend_elements, loc='best')

                cursor = mplcursors.cursor(ax[count], hover=True)
                if count == 0:
                    cursor.connect("add", lambda sel: sel.annotation.set_text(round(df_jawac.proba[int(sel.index)], 4)))
                else:
                    cursor.connect("add", lambda sel: sel.annotation.set_text(round(df_jawac.pred_proba[int(sel.index)], 4)))
                count+=1

            plt.show()
        else:
            raise Exception('mk3 or edf file path does not exist')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_directory', type=str, default='', help='directory path with analysis to use for testing neural network and finding residuals')
    parser.add_argument('--model_directory', type=str, default='', help='directory path with model to use')
    parser.add_argument('--model_name', type=str, default='LSTM', help='learning model architecture - either CNN or LSTM')
    parser.add_argument('--task', type=int, default='1', help='corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep')
    return parser.parse_args()


def main(p):
    find_residuals(**vars(p))
    print('-------- DONE --------')


if __name__ == '__main__':
    # statement to avoid useless warnings during training
    # export TF_CPP_MIN_LOG_LEVEL=3
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    opt = parse_opt()
    main(p=opt)

    # Test cmd lines
    # python code/training/residuals.py --analysis_directory 'C:\Users\cdetr\Documents\GitHub\nomics\jawac_processing_nomics\data\valid_invalid_analysis'  --model_name 'LSTM' --task 1
    # python code/training/residuals.py --analysis_directory 'C:\Users\cdetr\Documents\GitHub\nomics\jawac_processing_nomics\data\awake_sleep_analysis_update'  --model_name 'LSTM' --task 2
