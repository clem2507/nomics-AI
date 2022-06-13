import os
import shutil
import numpy as np

if __name__ == '__main__':
    with_oe_dir = '/Users/clemdetry/Documents/GitHub/nomics/nomics-AI/data/awake_sleep/tfe/with-oe/analysis'
    without_oe_dir = '/Users/clemdetry/Documents/GitHub/nomics/nomics-AI/data/awake_sleep/tfe/without-oe/analysis'
    files_with_oe = sorted(os.listdir(with_oe_dir))
    files_without_oe = sorted(os.listdir(without_oe_dir))
    for f in files_with_oe:
        if not f.startswith('.'):
            if f in files_without_oe:
                mk3_with_oe = open(f'{with_oe_dir}/{f}/{f}.mk3', 'a')
                mk3_without_oe = open(f'{without_oe_dir}/{f}/{f}.mk3')
                lines_without_oe = mk3_without_oe.readlines()
                temp_lines = []
                for l in lines_without_oe:
                    temp = l.split(';')
                    if (temp[len(temp) - 1]).strip() == 'W':
                        temp_lines.append(l)
                if len(temp_lines) > 0:
                    for line in temp_lines:
                        mk3_with_oe.write(f'{line}')
            else:
                shutil.rmtree(f'{with_oe_dir}/{f}')
