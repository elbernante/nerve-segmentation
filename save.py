from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os, glob

from shutil import copy, copytree, ignore_patterns
from lib.file_io import create_if_not_exists
from lib.file_io import latest_checkpoint_index_getter

from config import CHECK_POINT_DIR, CHECK_POINT_FILE

get_latest_checkpoint_index = latest_checkpoint_index_getter(CHECK_POINT_DIR,
                                                             CHECK_POINT_FILE)

_RUNS_DIR = 'runs'
def get_last_run():
    runs_dir = create_if_not_exists(_RUNS_DIR)
    runs = glob.glob(os.path.join(runs_dir, '*'))
    runs = [int(os.path.basename(r)) for r in runs]
    return 0 if len(runs) == 0 else max(runs)

def save_run(run_id):
    dst = create_if_not_exists(os.path.join(_RUNS_DIR, str(run_id)))
    copy('config.py', dst)
    copy('engine.py', dst)
    copy('train.py', dst)
    copy('test.py', dst)
    copy('preprocess.py', dst)
    copy('plot_log.ipynb', dst)

    copytree('lib', os.path.join(dst, 'lib'),
             ignore=ignore_patterns('*.pyc', '__pycache__'))

    output_dir = os.path.join(dst, 'output')
    copytree('output', output_dir,
             ignore=ignore_patterns('checkpoints'))

    # Save only the latest checkpoint file
    chck_pnt_dir = create_if_not_exists(os.path.join(output_dir, 'checkpoints'))
    latest_checkpoint = get_latest_checkpoint_index()
    checkpoint_file = os.path.join(CHECK_POINT_DIR, 
                                   CHECK_POINT_FILE.format(latest_checkpoint))
    copy(checkpoint_file, chck_pnt_dir)
    copy(checkpoint_file + '.meta', chck_pnt_dir)

    print("Files saved to: '{}'".format(dst))

last_run = get_last_run()
save_run(last_run + 1)