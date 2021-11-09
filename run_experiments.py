import os

from deep_learning_models.cnn_model import train_cnn
from deep_learning_models.lstm_model import train_lstm
from deep_learning_models.cnn_model import hyperparameters_tuning as ht_cnn
from deep_learning_models.lstm_model import hyperparameters_tuning as ht_lstm

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    num_class = 2

    if num_class == 2:
        epochs = 5
    else:
        epochs = 10



    # from classify_jawac import analysis_classification
    # from random import shuffle
    # from tqdm import tqdm
    # directory = 'data/analysis'
    # dir_names = os.listdir(directory)
    # shuffle(dir_names)
    # for i in tqdm(range(len(dir_names))):
    #     if dir_names[i].startswith('2'):
    #         analysis_classification(edf=f'{directory}/{dir_names[i]}/{dir_names[i]}.edf', model='lstm', num_class=2)



    # import mne
    # import pandas as pd
    # from matplotlib import pyplot as plt
    # from tqdm import tqdm
    # from utils.util import extract_data_from_line, string_datetime_conversion, block_print, enable_print
    #
    # # d = '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/all_invalid_analysis'
    # d = '/Users/clemdetry/Documents/UM/Third year/Nomics Thesis/data/valid_analysis_rest'
    # direct = sorted(os.listdir(d))
    # direct = direct[::-1]
    # for i in tqdm(range(len(direct))):
    #     if direct[i].startswith('2'):
    #         # df_mk3 = pd.read_pickle(os.path.dirname(os.path.abspath('run_script.py')) + f'/data/dataset/valid_mk3_dfs/{direct[i]}')
    #         # df_jawac = pd.read_pickle(os.path.dirname(os.path.abspath('run_script.py')) + f'/data/dataset/valid_jawac_dfs/{direct[i][:-8]}_jawac.pkl')
    #
    #         mk3_file = os.path.join(d, direct[i] + '/' + direct[i] + '.mk3')
    #         data_mk3 = open(mk3_file)
    #         lines = data_mk3.readlines()
    #         start_record_time = lines[5].split(';')[0]
    #         start_record_date = lines[5].split(';')[1]
    #         lines = lines[7:]
    #         col_names = ['start', 'end', 'label']
    #         df_mk3 = pd.DataFrame(columns=col_names)
    #         for line in lines:
    #             if line.split(sep=';')[-1][:-1] == 'Out of Range':
    #                 if line.split(sep=';')[-2] == '1':
    #                     temp = pd.Series(extract_data_from_line(line), index=df_mk3.columns)
    #                     df_mk3 = df_mk3.append(temp, ignore_index=True)
    #             else:
    #                 temp = pd.Series(extract_data_from_line(line), index=df_mk3.columns)
    #                 df_mk3 = df_mk3.append(temp, ignore_index=True)
    #
    #         edf_file = os.path.join(d, direct[i] + '/' + direct[i] + '.edf')
    #         block_print()
    #         raw_data = mne.io.read_raw_edf(edf_file)
    #         enable_print()
    #         data, times = raw_data[:]
    #         sec_interval = 0.1
    #         times = string_datetime_conversion(times, start_record_time, start_record_date)
    #         df_jawac = pd.DataFrame()
    #         df_jawac.insert(0, 'times', times)
    #         df_jawac.insert(1, 'data', data[0])
    #         df_jawac = df_jawac.set_index('times')
    #
    #         # graph
    #         fig, ax = plt.subplots()
    #         fig.set_size_inches(18.5, 10.5)
    #
    #         # ax.plot(df_jawac.resample(str(graph_time_resampling)+'S').median()['data'].to_frame(name='data').index.tolist(), df_jawac.resample(str(graph_time_resampling)+'S').median()['data'].to_frame(name='data').data.tolist())
    #         ax.plot(df_jawac.index.tolist(), df_jawac.data.tolist())
    #         ax.axhline(y=0, color='r', linewidth=1)
    #         ax.set(xlabel='time', ylabel='opening (mm)', title=f'Jawac Signal - {direct[i]}')
    #         ax.grid()
    #
    #         for idx, row in df_mk3.iterrows():
    #             if row.label == "Zone d'exclusion":
    #                 plt.axvspan(row.start, row.end, facecolor='b', alpha=0.2)
    #
    #         # for idx, row in df_mk3.iterrows():
    #         #     if row.label == "Out of Range":
    #                 # plt.axvspan(row.start, row.end, facecolor='r', alpha=0.2)
    #
    #         plt.show()

    # os.path.abspath('classify_jawac.py')) + '/data/analysis'

    # train_cnn(time_split=1, time_resampling=0.1, epochs=epochs, num_class=num_class)
    # train_cnn(time_split=1, time_resampling=1, epochs=epochs, num_class=num_class)
    # train_cnn(time_split=1, time_resampling=3, epochs=epochs, num_class=num_class)

    # train_cnn(time_split=3, time_resampling=0.1, epochs=epochs, num_class=num_class)
    # train_cnn(time_split=3, time_resampling=1, epochs=epochs, num_class=num_class)
    # train_cnn(time_split=3, time_resampling=3, epochs=epochs, num_class=num_class)

    # train_cnn(time_split=5, time_resampling=1, epochs=epochs, num_class=num_class)

    # train_lstm(time_split=1, time_resampling=0.1, epochs=epochs, num_class=num_class)
    # train_lstm(time_split=1, time_resampling=1, epochs=epochs, num_class=num_class)
    # train_lstm(time_split=1, time_resampling=3, epochs=epochs, num_class=num_class)

    # train_lstm(time_split=3, time_resampling=0.1, epochs=epochs, num_class=num_class)
    # train_lstm(time_split=3, time_resampling=1, epochs=epochs, num_class=num_class)
    # train_lstm(time_split=3, time_resampling=3, epochs=epochs, num_class=num_class)

    # train_lstm(time_split=5, time_resampling=1, epochs=epochs, num_class=num_class)

    # ht_cnn(time_split=1, time_resampling=1, max_trials=10, epochs=epochs, batch_size=32, num_class=num_class)
    # test_sequences(model_path=os.path.dirname(os.path.abspath('classify_jawac.py')) + '/models/cnn/saved_model')

    # ht_lstm(time_split=1, time_resampling=1, max_trials=20, epochs=epochs, batch_size=32, num_class=num_class)
    # test_sequences(model_path=os.path.dirname(os.path.abspath('classify_jawac.py')) + '/models/lstm/saved_model')
