import os
import argparse

from data_loader.preprocessing import create_dataframes


def run(directory):
    full_path = os.path.dirname(os.path.abspath('classify_jawac.py')) + f'/{directory}'
    if os.path.exists(directory):
        create_dataframes(directory=full_path)
    else:
        raise Exception('input directory does not exist')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='', help='directory path with new analysis files to add')
    return parser.parse_args()


def main(p):
    run(**vars(p))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # python update_db.py --directory 'data/analysis'

    opt = parse_opt()
    main(p=opt)
