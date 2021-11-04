import os
import argparse

from utils.util import analysis_classification


def run(edf, model):
    analysis_classification(edf=edf, model=model)


def parse_opt():
    parser = argparse.ArgumentParser()
    # edf file path or object as input??
    parser.add_argument('--edf', type=str, default='', help='edf file path for time series extraction')
    parser.add_argument('--model', type=str, default='LSTM', help='deep learning model - either CNN or LSTM')
    return parser.parse_args()


def main(p):
    run(**vars(p))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # invalid
    # python run_script.py --edf 'data/invalid_analysis/2019_01_08_22_13_32_121-SER-15-407(R1)_FR_38y/2019_01_08_22_13_32_121-SER-15-407(R1)_FR_38y.edf' --model 'LSTM'
    # python run_script.py --edf 'data/invalid_analysis/2019_01_08_16_38_57_121-SER-14-369(R1)_FR_79y/2019_01_08_16_38_57_121-SER-14-369(R1)_FR_79y.edf' --model 'LSTM'
    # python run_script.py --edf 'data/invalid_analysis/2019_01_11_15_11_45_121-SER-14-335(R1)_FR/2019_01_11_15_11_45_121-SER-14-335(R1)_FR.edf' --model 'LSTM'
    # python run_script.py --edf 'data/invalid_analysis/2019_01_13_22_11_13_121-SER-14-346(R1)_FR_56y/2019_01_13_22_11_13_121-SER-14-346(R1)_FR_56y.edf' --model 'LSTM'
    # valid
    # python run_script.py --edf 'data/valid_analysis/2019_01_31_23_56_20_121-SER-14-372(R2)_FR/2019_01_31_23_56_20_121-SER-14-372(R2)_FR.edf' --model 'LSTM'
    # python run_script.py --edf 'data/valid_analysis/2019_01_30_00_55_05_121-SER-16-495(R1)_FR_69y/2019_01_30_00_55_05_121-SER-16-495(R1)_FR_69y.edf' --model 'LSTM'
    # python run_script.py --edf 'data/valid_analysis/2019_01_07_15_53_00_121-SER-10-130(R3)_FR_36y/2019_01_07_15_53_00_121-SER-10-130(R3)_FR_36y.edf' --model 'LSTM'
    # TO CHECK
    # python run_script.py --edf 'data/valid_analysis/2019_01_03_19_57_59_121-SER-16-463(R2)_NL/2019_01_03_19_57_59_121-SER-16-463(R2)_NL.edf' --model 'LSTM'
    opt = parse_opt()
    main(p=opt)
