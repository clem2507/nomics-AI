import os
import sys
import pickle
import argparse

from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')

from util import move_figure


def open_figures(analysis_directory):
    """
    Method used to open the residual plots already computed and saved

    Parameters:

    -analysis_directory: directory path with the saved residual figures computed on a certain analysis directory
    """

    if not analysis_directory[:-8] == '_edf_dfs':
        if not os.path.exists(f'{analysis_directory}_edf_dfs'):
            raise Exception(f'{analysis_directory}_edf_dfs', '-> path does not exist')
        else:
            analysis_directory = f'{analysis_directory}_edf_dfs'
    filenames = sorted(os.listdir(analysis_directory))
    for i in tqdm(range(len(filenames))):
        fig_path = f'{analysis_directory}/{filenames[i]}/residuals_plt.pickle'
        if os.path.exists(fig_path):
            fig = pickle.load(open(fig_path, 'rb'))
            move_figure(fig, 0, 0)
            plt.show()
        else:
            raise Exception(fig_path, '-> path does not exist')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_directory', type=str, default='', help='directory path with the saved residual figures computed on a certain analysis directory')
    return parser.parse_args()


def main(p):
    open_figures(**vars(p))
    print('-------- DONE --------')


if __name__ == '__main__':
    # statement to avoid useless warnings during training
    # export TF_CPP_MIN_LOG_LEVEL=3
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    opt = parse_opt()
    main(p=opt)

    # Test cmd lines
    # python code/testing/residuals/open.py --analysis_directory '.\data\valid_invalid_analysis'
    # python code/testing/residuals/open.py --analysis_directory '.\data\awake_sleep_analysis_update'
