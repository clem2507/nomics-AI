import os
import time
import argparse

# from deep_learning_models.cnn_model import train_cnn
# from deep_learning_models.lstm_model import train_lstm
# from deep_learning_models.cnn_model import hyperparameters_tuning as ht_cnn
# from deep_learning_models.lstm_model import hyperparameters_tuning as ht_lstm

from utils.util import f1_m, analysis_classification


def run(edf, model):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start = time.time()
    analysis_classification(edf=edf, model=model)
    end = time.time()
    print('execution time =', (end-start), 'sec')
    print('--------')


def parse_opt():
    parser = argparse.ArgumentParser()
    # edf file path or object as input??
    parser.add_argument('--edf', type=str, default='', help='edf file path for time series extraction')
    parser.add_argument('--model', type=str, default='LSTM', help='deep learning model - either CNN or LSTM')
    return parser.parse_args()


def main(p):
    run(**vars(p))


if __name__ == '__main__':
    # python runscript.py --edf 'data/all_analysis/2019_01_08_22_13_32_121-SER-15-407(R1)_FR_38y/2019_01_08_22_13_32_121-SER-15-407(R1)_FR_38y.edf' --model 'LSTM'
    # python runscript.py --edf 'data/all_analysis/2019_01_30_00_55_05_121-SER-16-495(R1)_FR_69y/2019_01_30_00_55_05_121-SER-16-495(R1)_FR_69y.edf' --model 'LSTM'
    # python runscript.py --edf 'data/all_analysis/2019_01_31_23_56_20_121-SER-14-372(R2)_FR/2019_01_31_23_56_20_121-SER-14-372(R2)_FR.edf' --model 'LSTM'
    opt = parse_opt()
    main(opt)

    # train_cnn(load_flag=True, create_flag=False)
    # ht_lstm()
    # test_sequences(model_path=os.path.dirname(os.path.abspath('runscript.py')) + '/models/cnn/saved_model')

    # train_lstm(load_flag=True, create_flag=False)
    # ht_cnn()
    # test_sequences(model_path=os.path.dirname(os.path.abspath('runscript.py')) + '/models/lstm/saved_model')
