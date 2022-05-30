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
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/training')
sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/classification')

from data_loader import DataLoader
from classify import analysis_classification


def find_residuals(analysis_directory, 
                   model_name,
                   show_graph, 
                   wandb_log):
    """
    Method used to find the residuals between the model predictions and the actual labels

    Parameters:

    -analysis_directory: directory path with analysis to use for testing neural network and finding residuals
    -model_name: learning model architecture - either MLP, CNN, ResNet or LSTM
    -show_graph: true to show the output graph
    -wandb_log: true to log the residual plots on weights and biases
    """

    if wandb_log:
        wandb.init(project="nomics-AI", entity="nomics")

    model_name = model_name.lower()

    saved_dir_task1 = os.path.dirname(os.path.abspath('util.py')) + f'/models/task1/{model_name}'
    best_dir = f'{saved_dir_task1}/best-1.0'
    model_list = os.listdir(best_dir)

    print('----- TASK 1 MODEL CHOICE -----')
    for i in range(len(model_list)):
        print(f'({i+1}) {model_list[i]}')
    model_num = int(input('model number: '))
    print('--> task 1 model name:', model_list[model_num-1])
    if model_num > 0 and model_num <= len(model_list):
        model_path_task1 = f'{best_dir}/{model_list[model_num-1]}/best/model-best.h5'
    else:
        most_recent_folder_path = sorted(Path(best_dir).iterdir(), key=os.path.getmtime)[::-1]
        most_recent_folder_path = [name for name in most_recent_folder_path if not (str(name).split('/')[-1]).startswith('.')]
        model_path_task1 = str(most_recent_folder_path[0]) + '/best/model-best.h5'

    saved_dir_task2 = os.path.dirname(os.path.abspath('util.py')) + f'/models/task2/{model_name}'
    best_dir = f'{saved_dir_task2}/best-1.0'
    model_list = os.listdir(best_dir)

    print('----- TASK 2 MODEL CHOICE -----')
    for i in range(len(model_list)):
        print(f'({i+1}) {model_list[i]}')
    model_num = int(input('model number: '))
    print('--> task 1 model name:', model_list[model_num-1])
    if model_num > 0 and model_num <= len(model_list):
        model_path_task2 = f'{best_dir}/{model_list[model_num-1]}/best/model-best.h5'
    else:
        most_recent_folder_path = sorted(Path(best_dir).iterdir(), key=os.path.getmtime)[::-1]
        most_recent_folder_path = [name for name in most_recent_folder_path if not (str(name).split('/')[-1]).startswith('.')]
        model_path_task2 = str(most_recent_folder_path[0]) + '/best/model-best.h5'
    print('------------------------')

    if os.path.exists(analysis_directory):
        DataLoader(analysis_directory=analysis_directory).create_dataframes()
    else:
        raise Exception('input directory does not exist')

    filenames = sorted(os.listdir(analysis_directory))
    res_data_total = 0
    for file_i in tqdm(range(len(filenames))):
        mk3_file = f'{analysis_directory}_edf_dfs/{filenames[file_i]}/{filenames[file_i]}_mk3.pkl'
        edf_file = f'{analysis_directory}_edf_dfs/{filenames[file_i]}/{filenames[file_i]}_jawac.pkl'
        df_mk3 = pd.read_pickle(mk3_file)
        df_jawac = pd.read_pickle(edf_file)

        df_jawac = df_jawac.resample(f'1S').median()

        df_jawac['label'] = [1 for n in range(len(df_jawac))]
        df_jawac['pred_label'] = [1 for n in range(len(df_jawac))]
        df_jawac['res_label'] = [1 for n in range(len(df_jawac))]
        df_jawac['proba'] = [1 for n in range(len(df_jawac))]
        df_jawac['pred_proba'] = [0.5 for n in range(len(df_jawac))]
        df_jawac['res_proba'] = [1 for n in range(len(df_jawac))]

        df_jawac.index = pd.to_datetime(df_jawac.index).tz_localize(None)

        for idx, row in df_mk3.iterrows():
            mask = (df_jawac.index >= row.start) & (df_jawac.index <= row.end)
            if row.label == 'W':
                df_jawac.loc[mask, ['label']] = 2
            elif row.label == 'Out of Range':
                df_jawac.loc[mask, ['label']] = 0

        df_jawac.reset_index(inplace=True)

        out_dic = analysis_classification(edf=f'{analysis_directory}/{filenames[file_i]}/{filenames[file_i]}.edf', 
                                          model=model_name, 
                                          model_path_task1=model_path_task1, 
                                          model_path_task2=model_path_task2, 
                                          stop_print=True)

        df_jawac.pred_label = out_dic['df_jawac'].label.values
        df_jawac.pred_proba = out_dic['df_jawac'].proba.values
        df_jawac.pred_proba = df_jawac.pred_proba.round(3)

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

        if 'hypno' in df_jawac.columns.values.tolist():
            graph_1_title = 'hypnogram'
        else:
            graph_1_title = 'labels'

        fig = make_subplots(rows=3, 
                            cols=1,
                            subplot_titles=(
                                f'Jawac Signal - {filenames[file_i]} - {graph_1_title}',
                                f'Jawac Signal - {filenames[file_i]} - predictions', 
                                f'Jawac Signal - {filenames[file_i]} - residuals'),
                            shared_xaxes=True)

        if 'hypno' in df_jawac.columns.values.tolist():
            fig.add_trace(
                go.Scatter(x=df_jawac.index.tolist(), 
                        y=df_jawac.hypno.tolist(), 
                        showlegend=False, 
                        hoverinfo='skip',
                        line=dict(
                            color='rgb(57, 119, 175)',
                            width=2)),
                row=1, col=1
            )
        else:
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
                       y=df_jawac.pred_label.tolist(), 
                       showlegend=False,
                       customdata = df_jawac.pred_proba.tolist(),
                       hovertemplate='<extra><br>proba: %{customdata}</extra>',
                       hoverinfo='skip',
                       line=dict(
                        color='rgb(57, 119, 175)',
                        width=2)),
            row=3, col=1
        )

        fig.update_layout(
            yaxis = dict(
                tickmode = 'array',
                tickvals = [-1, 0, 1, 2, 3, 4],
                ticktext = ['Inv', 'REM', 'S3', 'S2', 'S1', 'Awake']
            ),
            yaxis3 = dict(
                tickmode = 'array',
                tickvals = [0, 1, 2],
                ticktext = ['Inv', 'Sleep', 'Awake']
            )
        )

        if 'hypno' in df_jawac.columns.values.tolist():
            count = 2
            df_list = [df_pred_label, df_residuals]
        else:
            df_list =  [df_label, df_pred_label, df_residuals]
            count = 1

        for df in df_list:
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

        if 'hypno' not in df_jawac.columns.values.tolist():
            fig.add_hline(y=0, row=1, col=1, line_color='red', line_width=1)
        fig.add_hline(y=0, row=2, col=1, line_color='red', line_width=1)

        fig.update_xaxes(title_text="time", row=1, col=1)
        fig.update_xaxes(title_text="time", row=2, col=1)
        fig.update_xaxes(title_text="time", row=3, col=1)

        if 'hypno' in df_jawac.columns.values.tolist():
            fig.update_yaxes(title_text="stade", row=1, col=1)
        else:
            fig.update_yaxes(title_text="opening (mm)", row=1, col=1)
        fig.update_yaxes(title_text="opening (mm)", row=2, col=1)
        fig.update_yaxes(title_text="opening (mm)", row=3, col=1)

        legend = 'green(sleep), red(invalid), blue(awake), yellow(residuals)'

        fig.update_layout(title=go.layout.Title(
                              text=f'Residual subplots <br><sup>Legend: {legend}</sup>',
                              xref='paper',
                              x=0))

        fig.layout.width = None
        fig.layout.height = None

        if wandb_log:
            wandb.log({'residuals_plt': fig})
        if show_graph:
            fig.show()

    res_min_total = (res_data_total) / 60
    res_hours_total = (res_data_total) / 3600
    print('average residual min:', res_min_total/len(filenames))
    print('average residual hours:', res_hours_total/len(filenames))
    if wandb_log:
        wandb.log({'average residual min': res_min_total/len(filenames),
                'average residual hours': res_hours_total/len(filenames),
                'model path task 1': model_path_task1,
                'model path task 2': model_path_task2})


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_directory', type=str, default='', help='directory path with analysis to use for testing neural network and finding residuals')
    parser.add_argument('--model_name', type=str, default='LSTM', help='learning model architecture - either MLP, CNN, ResNet or LSTM')
    parser.add_argument('--show_graph', dest='show_graph', action='store_true', help='invoke to show the output graph')
    parser.add_argument('--wandb_log', dest='wandb_log', action='store_true', help='invoke to log the residual plots on weights and biases')
    parser.set_defaults(show_graph=False)
    parser.set_defaults(wandb_log=False)
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
    # python code/testing/residuals/find.py --analysis_directory 'data/valid_invalid/analysis' --model_name 'LSTM' --show_graph
    # python code/testing/residuals/find.py --analysis_directory 'data/awake_sleep/analysis' --model_name 'LSTM' --show_graph
    # python code/testing/residuals/find.py --analysis_directory 'data/awake_sleep/jawrhin/analysis' --model_name 'LSTM' --show_graph

    # Test cmd lines MACOS
    # python3 code/testing/residuals/find.py --analysis_directory '/Users/clemdetry/Documents/GitHub/nomics/jawac_processing_nomics/data/valid_invalid_analysis'  --model_name 'LSTM' --task 1 --show_graph
    # python3 code/testing/residuals/find.py --analysis_directory '/Users/clemdetry/Documents/GitHub/nomics/jawac_processing_nomics/data/awake_sleep_analysis'  --model_name 'LSTM' --task 2 --show_graph
