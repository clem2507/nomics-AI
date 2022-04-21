import os
import sys
import pickle
import argparse

from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath('util.py')) + '/code/utils')

from util import move_figure


def open_figures(model_path):
    """
    Method used to open the residual plots already computed and saved

    Parameters:

    -model_path: model path with the saved residual figures computed on a certain analysis directory
    """

    if not os.path.exists(f'{model_path}/residuals'):
        raise Exception(f'{model_path}/residuals', '-> path does not exist')
    filenames = sorted(os.listdir(f'{model_path}/residuals'))
    for i in tqdm(range(len(filenames))):
        fig_path = f'{model_path}/residuals/{filenames[i]}/residuals_plt.pickle'
        if os.path.exists(fig_path):
            fig = pickle.load(open(fig_path, 'rb'))
            move_figure(fig, 0, 0)
            plt.show()
        else:
            raise Exception(fig_path, '-> path does not exist')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='', help='model path with the saved residual figures computed on a certain analysis directory')
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

    # Test cmd lines - WINDOWS
    # python code/testing/residuals/open.py --analysis_directory '.\data\valid_invalid_analysis'
    # python code/testing/residuals/open.py --analysis_directory '.\data\awake_sleep_analysis_update'

    # Test cmd lines - MACOS
    # python3 code/testing/residuals/open.py --analysis_directory '/Users/clemdetry/Documents/GitHub/nomics/jawac_processing_nomics/data/valid_invalid_analysis'
    # python3 code/testing/residuals/open.py --analysis_directory '/Users/clemdetry/Documents/GitHub/nomics/jawac_processing_nomics/data/awake_sleep_analysis_update'
