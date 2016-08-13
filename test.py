from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os, glob

from train import run_test
from shutil import copy

from config import CHECK_POINT_DIR, CHECK_POINT_FILE, HIGHEST_DIR

def get_index_of_highest_checkpoint():
    cp_files_highest = glob.glob(os.path.join(HIGHEST_DIR, 
                                      CHECK_POINT_FILE.format('*')))
    if len(cp_files_highest) == 0:
        return 'latest'

    index = int(cp_files_highest[0].split('-')[-1].split('.')[0])

    # Check if checkpoint files exists in the CHECK_POINT_DIR directory
    check_points = [os.path.basename(f) for f in cp_files_highest]
    cp_files_cpdir = [os.path.join(CHECK_POINT_DIR, f) for f in check_points]
    exists = all([os.path.isfile(f) for f in cp_files_cpdir])

    # If it doesn't already exists, copy from HIGHEST_DIR
    if not exists:
        for f in cp_files_highest:
            copy(f, CHECK_POINT_DIR)

    return index

checkpoint_index = get_index_of_highest_checkpoint()
run_test(checkpoint_index=checkpoint_index)