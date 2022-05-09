import wandb

import os
import sys
import mne
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go

from tqdm import tqdm
from pathlib import Path
from keras.models import load_model
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')
sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/training')

from util import f1, get_value_in_line, extract_data_from_line, block_print, enable_print, datetime_conversion
from data_loader import DataLoader


def find_residuals(analysis_directory, 
                   model_name, 
                   model_num, 
                   task, 
                   show_graph, 
                   save_graph):
    """
    Method used to find the residuals between the model predictions and the actual labels

    Parameters:

    -analysis_directory: directory path with analysis to use for testing neural network and finding residuals
    -model_name: learning model architecture - either CNN or LSTM
    -model_num: number of the model to use in the best folder
    -task: corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep
    -show_graph: true to show the output graph
    -save_graph: true to save the output residual plots
    """

    if save_graph:
        wandb.init(project="nomics-AI", entity="nomics")

    model_name = model_name.lower()
    saved_dir = os.path.dirname(os.path.abspath('util.py')) + f'/models/task{task}/{model_name}'
    best_dir = f'{saved_dir}/best'
    model_list = os.listdir(best_dir)

    print('----- MODEL CHOICE -----')
    for i in range(len(model_list)):
        print(f'({i+1}) {model_list[i]}')
    if model_num is None:
        model_num = int(input('model number: '))
    else:
        print('model number:', model_num)
    print('--> model name:', model_list[model_num-1])
    if model_num > 0 and model_num <= len(model_list):
        model_path = f'{best_dir}/{model_list[model_num-1]}/best/model-best.h5'
        info_path = f'{best_dir}/{model_list[model_num-1]}/info.txt'
    else:
        most_recent_folder_path = sorted(Path(best_dir).iterdir(), key=os.path.getmtime)[::-1]
        most_recent_folder_path = [name for name in most_recent_folder_path if not (str(name).split('/')[-1]).startswith('.')]
        model_path = str(most_recent_folder_path[0]) + '/best/model-best.h5'
        info_path = str(most_recent_folder_path[0]) + '/info.txt'
    print('------------------------')

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
    if not analysis_directory[:-8] == '_edf_dfs':
        if os.path.exists(f'{analysis_directory}_edf_dfs'):
            analysis_directory = f'{analysis_directory}_edf_dfs'
        else:
            print('WARNING -- edf_dfs directory path does not exist')
            if os.path.exists(analysis_directory):
                DataLoader(analysis_directory=analysis_directory, task=task).create_dataframes()
            else:
                raise Exception(analysis_directory, '-> analysis directory path does not exist')
    # filenames = sorted(os.listdir(analysis_directory))
    filenames = os.listdir(analysis_directory)
    # random.shuffle(filenames)
    res_data_total = 0
    for file_i in tqdm(range(len(filenames))):
        mk3_file = f'{analysis_directory}/{filenames[file_i]}/{filenames[file_i]}_mk3.pkl'
        edf_file = f'{analysis_directory}/{filenames[file_i]}/{filenames[file_i]}_jawac.pkl'
        if not os.path.exists(mk3_file) and not os.path.exists(edf_file):
            mk3_file = f'{analysis_directory}/{filenames[file_i]}/{filenames[file_i]}.mk3'
            edf_file = f'{analysis_directory}/{filenames[file_i]}/{filenames[file_i]}.edf'
            if os.path.exists(mk3_file) and os.path.exists(edf_file):
                data_mk3 = open(mk3_file)    # mk3 file reading
                lines = data_mk3.readlines()    # list with the mk3 lines
                if task == 1:
                    lines = lines[7:]    # log file information
                elif task == 2:
                    lines = lines[5:]    # log file information
                col_names = ['start', 'end', 'label']
                df_mk3 = pd.DataFrame(columns=col_names)    # dataframe with label data created
                for line in lines:
                    if len(line) > 0:
                        temp = pd.DataFrame(extract_data_from_line(line), columns=df_mk3.columns)
                        df_mk3 = pd.concat([df_mk3, temp], ignore_index=True)
                    
                block_print()
                raw_data = mne.io.read_raw_edf(edf_file)    # edf file reading
                enable_print()

                df_jawac = pd.DataFrame()    # dataframe creation to store time series data
                if task == 1:
                    data, times = raw_data[:]    # edf file data extraction
                    times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])    # conversion to usable date dtype 
                    df_jawac.insert(0, 'times', times)
                    df_jawac.insert(1, 'data', data[0])
                    df_jawac = df_jawac.resample('0.1S', on='times').median()
                elif task == 2:
                    data = raw_data[0][0][0]
                    times = raw_data[1][1]
                    times = datetime_conversion(times, raw_data.__dict__['info']['meas_date'])    # conversion to usable date dtype 
                    df_jawac.insert(0, 'times', times)
                    df_jawac.insert(1, 'data', data)
                    df_jawac = df_jawac.resample('0.1S', on='times').median()
            else:
                raise Exception(mk3_file, 'and', edf_file, ' -> path does not exist')
        else:
            df_mk3 = pd.read_pickle(mk3_file)
            df_jawac = pd.read_pickle(edf_file)

        df_jawac = df_jawac.resample(f'{downsampling_value}S').median()
        df_jawac['label'] = [1 for n in range(len(df_jawac))]
        df_jawac['pred_label'] = [1 for n in range(len(df_jawac))]
        df_jawac['res_label'] = [1 for n in range(len(df_jawac))]
        df_jawac.index = pd.to_datetime(df_jawac.index).tz_localize(None)
        for idx, row in df_mk3.iterrows():
            mask = (df_jawac.index >= row.start) & (df_jawac.index <= row.end)
            if task == 1:
                df_jawac.loc[mask, ['label']] = 0
            else:
                if row.label == 'W':
                    df_jawac.loc[mask, ['label']] = 2
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
                    if task == 1:
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
            data = df_jawac.data.tolist()
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

        df_jawac['proba'] = [1 for n in range(len(df_jawac))]
        df_jawac['pred_proba'] = [0.5 for n in range(len(df_jawac))]
        df_jawac['res_proba'] = [1 for n in range(len(df_jawac))]

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

        df_jawac.pred_proba = df_jawac.pred_proba.round(3)

        min_gap_time = int((60 / downsampling_value) * 1)   # in minutes
        previous_label = df_jawac.pred_label[0]
        idx = []
        for i in range(len(df_jawac.pred_label)):
            idx.append(i)
            if df_jawac.pred_label[i] != previous_label:
                if len(idx) <= min_gap_time:
                    if task == 1:
                        if previous_label == 0:
                            df_jawac.loc[idx, 'pred_label'] = 1
                        elif previous_label == 1:
                            df_jawac.loc[idx, 'pred_label'] = 0
                    elif task == 2:
                        if previous_label == 1:
                            df_jawac.loc[idx, 'pred_label'] = 2
                        elif previous_label == 2:
                            df_jawac.loc[idx, 'pred_label'] = 1
                    df_jawac.loc[idx, 'pred_proba'] = 0.5
                else:
                    idx = []
                    idx.append(i)
            previous_label = df_jawac.pred_label[i]

        for idx, row in df_jawac.iterrows():
            if row.label != row.pred_label:
                df_jawac.loc[idx, ['res_label']] = 3

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
                if row.pred_label != saved_label:
                    temp = pd.DataFrame(data = [[saved_start, row.times, saved_label, np.mean(proba), len(proba)]], columns=['start', 'end', 'label', 'proba', 'data_num'])
                    df_pred_label = pd.concat([df_pred_label, temp], ignore_index=True)
                    saved_label = row.pred_label
                    saved_start = row.times
                    proba = []
        if len(proba) > 0:
            temp = pd.DataFrame(data = [[saved_start, df_jawac.times.tolist()[-1], saved_label, round(np.mean(proba), 3), len(proba)]], columns=['start', 'end', 'label', 'proba', 'data_num'])
            df_pred_label = pd.concat([df_pred_label, temp], ignore_index=True)

        df_residuals = pd.DataFrame(columns=['start', 'end', 'label', 'proba', 'data_num'])
        saved_label = df_jawac.iloc[0].res_label
        saved_start = df_jawac.iloc[0].times
        proba = []
        for idx, row in df_jawac.iterrows():
            if len(row) > 0:
                proba.append(row.res_proba)
                if row.res_label != saved_label:
                    if saved_label == 3:
                        temp = pd.DataFrame(data = [[saved_start, row.times, saved_label, np.mean(proba), len(proba)]], columns=['start', 'end', 'label', 'proba', 'data_num'])
                        df_residuals = pd.concat([df_residuals, temp], ignore_index=True)
                    saved_label = row.res_label
                    saved_start = row.times
                    proba = []
        if len(proba) > 0:
            if saved_label == 3:
                temp = pd.DataFrame(data = [[saved_start, df_jawac.times.tolist()[-1], saved_label, np.mean(proba), len(proba)]], columns=['start', 'end', 'label', 'proba', 'data_num'])
                df_residuals = pd.concat([df_residuals, temp], ignore_index=True)

        if 3 in df_jawac.res_label.value_counts().keys():
            res_data_total += df_jawac.res_label.value_counts()[3]

        df_jawac.set_index('times', inplace=True)

        fig = make_subplots(rows=3, 
                            cols=1,
                            subplot_titles=(
                                f'Jawac Signal - {filenames[file_i]} - actual', 
                                f'Jawac Signal - {filenames[file_i]} - predicted', 
                                f'Jawac Signal - {filenames[file_i]} - residuals'))

        fig.add_trace(
            go.Scatter(x=df_jawac.index.tolist(), 
                       y=df_jawac.data.tolist(), 
                       showlegend=False, 
                       hoverinfo='skip',
                       line=dict(
                        color='rgb(57, 119, 175)',
                        width=2)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df_jawac.index.tolist(), 
                       y=df_jawac.data.tolist(), 
                       showlegend=False,
                       customdata = df_jawac.pred_proba.tolist(),
                       hovertemplate='<extra><br>proba: %{customdata}</extra>',
                       hoverinfo='skip',
                       line=dict(
                        color='rgb(57, 119, 175)',
                        width=2)),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=df_jawac.index.tolist(), 
                       y=df_jawac.data.tolist(), 
                       showlegend=False,
                       hoverinfo='skip',
                       line=dict(
                        color='rgb(57, 119, 175)',
                        width=2)),
            row=3, col=1
        )

        count = 1
        for df in [df_label, df_pred_label, df_residuals]:
            for idx, row in df.iterrows():
                if row.label == 0:
                    fig.add_vrect(x0=row.start, x1=row.end,
                                  fillcolor='red', 
                                  opacity=0.2*row.proba,
                                  line_width=0,
                                  row=count, col=1)
                elif row.label == 1:
                    fig.add_vrect(x0=row.start, x1=row.end,
                                  fillcolor='green', 
                                  opacity=0.2*row.proba,
                                  line_width=0,
                                  row=count, col=1)
                elif row.label == 2:
                    fig.add_vrect(x0=row.start, x1=row.end,
                                  fillcolor='blue', 
                                  opacity=0.2*row.proba,
                                  line_width=0,
                                  row=count, col=1)
                elif row.label == 3:
                    fig.add_vrect(x0=row.start, x1=row.end,
                                  fillcolor='yellow', 
                                  opacity=0.3,
                                  line_width=0,
                                  row=count, col=1)
            count+=1

        fig.add_hline(y=0, row=1, col=1, line_color='red', line_width=1)
        fig.add_hline(y=0, row=2, col=1, line_color='red', line_width=1)
        fig.add_hline(y=0, row=3, col=1, line_color='red', line_width=1)

        fig.update_xaxes(title_text="time", row=1, col=1)
        fig.update_xaxes(title_text="time", row=2, col=1)
        fig.update_xaxes(title_text="time", row=3, col=1)

        fig.update_yaxes(title_text="opening (mm)", row=1, col=1)
        fig.update_yaxes(title_text="opening (mm)", row=2, col=1)
        fig.update_yaxes(title_text="opening (mm)", row=3, col=1)

        if task == 1:
            legend = 'red(invalid), green(valid), yellow(residuals)'
        elif task == 2:
            legend = 'green(sleep), blue(awake), yellow(residuals)'

        fig.update_layout(title=go.layout.Title(
                              text=f'Residual subplots <br><sup>Legend: {legend}</sup>',
                              xref='paper',
                              x=0))

        fig.layout.width = None
        fig.layout.height = None
        
        if save_graph:
            if not os.path.exists(f'{best_dir}/{model_list[model_num-1]}/residuals'):
                os.mkdir(f'{best_dir}/{model_list[model_num-1]}/residuals')
            if not os.path.exists(f'{best_dir}/{model_list[model_num-1]}/residuals/{filenames[file_i]}'):
                os.mkdir(f'{best_dir}/{model_list[model_num-1]}/residuals/{filenames[file_i]}')
            fig.write_image(f'{best_dir}/{model_list[model_num-1]}/residuals/{filenames[file_i]}/residuals_plt.png')
            wandb.log({'residuals_plt': fig})
        if show_graph:
            fig.show()

    res_min_total = (res_data_total * (1/downsampling_value)) / 60
    res_hours_total = (res_data_total * (1/downsampling_value)) / 3600
    print('average residual min:', res_min_total/len(filenames))
    print('average residual hours:', res_hours_total/len(filenames))
    if save_graph:
        wandb.log({'average residual min': res_min_total/len(filenames),
                'average residual hours': res_hours_total/len(filenames)})


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_directory', type=str, default='', help='directory path with analysis to use for testing neural network and finding residuals')
    parser.add_argument('--model_name', type=str, default='LSTM', help='learning model architecture - either CNN or LSTM')
    parser.add_argument('--model_num', type=int, default=None, help='number of the model to use in the best folder')
    parser.add_argument('--task', type=int, default='1', help='corresponding task number, so far: task = 1 for valid/invalid and task = 2 for awake/sleep')
    parser.add_argument('--show_graph', dest='show_graph', action='store_true', help='invoke to show the output graph')
    parser.add_argument('--save_graph', dest='save_graph', action='store_true', help='invoke to save the output residual plots')
    parser.set_defaults(show_graph=False)
    parser.set_defaults(save_graph=False)
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

    # Test cmd lines WINDOWS
    # python code/testing/residuals/find.py --analysis_directory '.\data\valid_invalid_analysis'  --model_name 'LSTM' --task 1 --show_graph
    # python code/testing/residuals/find.py --analysis_directory '.\data\awake_sleep_analysis_update'  --model_name 'LSTM' --task 2 --show_graph

    # Test cmd lines MACOS
    # python3 code/testing/residuals/find.py --analysis_directory '/Users/clemdetry/Documents/GitHub/nomics/jawac_processing_nomics/data/valid_invalid_analysis'  --model_name 'LSTM' --task 1 --show_graph
    # python3 code/testing/residuals/find.py --analysis_directory '/Users/clemdetry/Documents/GitHub/nomics/jawac_processing_nomics/data/awake_sleep_analysis_update'  --model_name 'LSTM' --task 2 --show_graph
