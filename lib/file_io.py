from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os, glob
from shutil import copy
import csv, operator

from six import PY3
from six.moves import range, zip, cPickle

def create_if_not_exists(dir_name):
    """Creates a directory if it doesn't already exists"""

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return dir_name

def del_files(path):
    """Deletes all in a given path"""
    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f):
            os.remove(f)

def _write_csv(f, header, data):
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

def _read_csv(f, li, has_header):
    reader = csv.reader(f)
    header = next(reader) if has_header else None
    for row in reader:
        li.append(row)

def _py2_write_to_csv(filename, header, data):
    with open(filename, "wb") as f:
        _write_csv(f, header, data)

def _py3_write_to_csv(filename, header, data):
    with open(filename, "w", newline='') as f:
        _write_csv(f, header, data)

def _py2_read_from_csv(filename, has_header=True):
    li = []
    with open(filename, "rb") as f:
        _read_csv(f, li, has_header)
    return li

def _py3_read_from_csv(filename, has_header=True):
    li = []
    with open(filename, "r", newline='') as f:
        _read_csv(f, li, has_header)
    return li

write_to_csv = _py3_write_to_csv if PY3 else _py2_write_to_csv
read_from_csv = _py3_read_from_csv if PY3 else _py2_read_from_csv


def loss_log_saver(loss_log_dir, loss_log_file):
    def save_loss_log(loss_log, epoch):
        f_name = os.path.join(loss_log_dir, loss_log_file.format(epoch))
        write_to_csv(f_name, ['loss', 'f1_score'], loss_log)
    return save_loss_log

def save_f_stat_log(f_stat_file, f_stats, epoch):
    f_name = f_stat_file.format(epoch)
    write_to_csv(f_name, 
                 ['true_positive', 'false_positive', 
                  'false_negative', 'true_negative'],
                 f_stats)

training_log_cache = {}
def get_training_log(log_file):
    if len(training_log_cache) == 0 \
            and os.path.isfile(log_file):
        val_types = [int, int] + [int] * 2 + [float] * 5
        for row in read_from_csv(log_file):
            vals = [conv(val) for conv, val in zip(val_types, row)]
            training_log_cache[vals[0]] = vals[1:]    
    return training_log_cache

def training_log_updater(log_file):
    def update_training_log(epoch, iterations, *args):
        assert len(args) == 7, """
            Expected arguments:
                - epoch
                - iterations
                - total training count
                - total validation count
                - learning rate
                - training loss
                - validation loss
                - training F1 score
                - validation F1 score
            """
                
        train_log = get_training_log(log_file)
        train_log[epoch] = [iterations] + list(args)
        sorted_acc = sorted(train_log.items(), key=operator.itemgetter(0))
        enum_log = [[e] + ai for e, ai in sorted_acc]
        header = ["epoch", "iters",
                  "train_count", "val_count",
                  "learn_rate",
                  "train_loss", "val_loss",
                  "train_f1", "val_f1"]
        write_to_csv(log_file, header, enum_log)
    return update_training_log

def latest_checkpoint_index_getter(checkpoint_dir, checkpoint_file):
    def get_latest_checkpoint_index():
        get_i = lambda f: int(f.split('-')[1].split('.')[0])
        cp_indexes = [get_i(f) for f in glob.glob(
                os.path.join(checkpoint_dir, checkpoint_file.format('*[0-9]')))]
        return -1 if len(cp_indexes) == 0 else max(cp_indexes)
    return get_latest_checkpoint_index

def checkpoint_getter(checkpoint_dir, checkpoint_file):
    def get_checkpoint_at_index(index):
        f_name = os.path.join(checkpoint_dir, checkpoint_file.format(index))
        if not os.path.isfile(f_name):
            raise IOError("Checkpoint file '{}' does not exist.".format(f_name))
        return f_name
    return get_checkpoint_at_index

def highest_checkpoint_saver(highest_dir,
                             highest_score_file,
                             checkpoint_dir,
                             checkpoint_file):
    def save_highest_checkpoint(score, epoch):
        # get highest
        highest_file = os.path.join(highest_dir, highest_score_file)
        if os.path.isfile(highest_file):
            with open(highest_file, 'rb') as f:
                highest = cPickle.load(f)
        else:
            highest = 0.0

        # save checkpiont if new score is higher than previous highest
        if score >= highest:
            del_files(os.path.join(highest_dir, '*'))
            cp_f = os.path.join(checkpoint_dir, checkpoint_file.format(epoch))
            cp_f_meta = cp_f + ".meta"
            copy(cp_f, highest_dir)
            copy(cp_f_meta, highest_dir)
            with open(highest_file, 'wb') as f:
                cPickle.dump(score, f)

    return save_highest_checkpoint