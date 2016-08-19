"""
Module for building and execution of tensor graph

@author:
    Peter James Bernante
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os, glob, time, timeit
from shutil import copy

import numpy as np
import tensorflow as tf

from math import ceil

from six import PY3, print_
from six.moves import range, zip, cPickle

from lib.variable_registry import memoise_variables, DEFAULT_REGISTRY_MANAGER
from lib.file_io import loss_log_saver, save_f_stat_log, training_log_updater
from lib.file_io import latest_checkpoint_index_getter, checkpoint_getter
from lib.file_io import highest_checkpoint_saver
from lib.stats_tools import get_f_stats, f_beta_score, np_f_beta_score
from lib.stats_tools import print_data_stats

from config import *

time_log = time.perf_counter if PY3 else timeit.default_timer
save_loss_log = loss_log_saver(LOSS_LOG_DIR, LOSS_LOG_FILE)
update_training_log = training_log_updater(TRAINING_LOG)
get_checkpoint_at_index = checkpoint_getter(CHECK_POINT_DIR, CHECK_POINT_FILE)
get_latest_checkpoint_index = latest_checkpoint_index_getter(CHECK_POINT_DIR,
                                                             CHECK_POINT_FILE)
save_highest_checkpoint = highest_checkpoint_saver(HIGHEST_DIR,
                                                   HIGHEST_SCORE_FILE,
                                                   CHECK_POINT_DIR,
                                                   CHECK_POINT_FILE)

comber = lambda o: tf.reshape(o, [-1, NUM_LABELS])
uncomber = lambda o: tf.reshape(o, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_LABELS])

def get_class_weights():
    """Returns class weights using median frequency balancing"""

    if not(os.path.isfile(TRAIN_STATS_PICKLE)):
        raise RuntimeError("Data not found. Run 'preprocess.py' first.")

    with open(TRAIN_STATS_PICKLE, 'rb') as f:
        stats = cPickle.load(f)
    median_freq = (stats['pos_freq'] + stats['neg_freq']) / 2.0
    pos = median_freq / stats['pos_freq']
    neg = median_freq / stats['neg_freq']
    scaled_pos = pos / max(pos, neg)
    scaled_neg = neg / max(pos, neg)
    return scaled_pos, scaled_neg

_class_weights = {'positive': None, 'negative': None}
def class_weights():
    """Returns a constant tensor conatinaing the class weights"""

    if _class_weights['positive'] is None or \
            _class_weights['negative'] is None:
       pos_class_weight, neg_class_weight = get_class_weights()
       _class_weights['positive'] = pos_class_weight
       _class_weights['negative'] = neg_class_weight

    return tf.constant([ _class_weights['negative'],
                         _class_weights['positive']])

def get_train_val_sets(image_dir=TRAIN_DIR):
    """Returns the train and validation dataset"""

    if not(os.path.isfile(TRAIN_SET_PICKLE) \
            and os.path.isfile(VALIDATION_SET_PICKLE)):
        raise RuntimeError("Data not found. Run 'preprocess.py' first.")

    print("\nReading data...")
    print("... loading {}".format(TRAIN_SET_PICKLE))
    t_set = np.load(TRAIN_SET_PICKLE)
    t_x = t_set['x']
    t_y = t_set['y']

    print("... loading {}".format(VALIDATION_SET_PICKLE))
    v_set = np.load(VALIDATION_SET_PICKLE)
    v_x = v_set['x']
    v_y = v_set['y']

    def verify_set(x, y):
        assert x.shape[1:] == (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), \
               "Unexpected image shape!"
        assert y.shape[1:] == (IMAGE_HEIGHT, IMAGE_WIDTH), \
               "Unexpected label shape!"
    
    verify_set(t_x, t_y)
    verify_set(v_x, v_y)
    
    return t_x, t_y, v_x, v_y

def loss_and_predict(logits, targets, l2):
    """Returns tensors for loss function and prediction"""

    combed_logits = comber(logits)
    weighted_logits = tf.mul(combed_logits, class_weights())
    cross_en = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        weighted_logits, tf.reshape(targets, [-1]))
    loss = tf.reduce_mean(cross_en) + l2
    pred = uncomber(tf.nn.softmax(combed_logits))
    return loss, pred

def prediction(logits):
    """Returns prediction tensor"""
    return uncomber(tf.nn.softmax(comber(logits)))

def optimize(loss):
    """Returns tensor for the optimizer for training"""

    # Learning rate
    global_epoch = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE, global_epoch, 
        DECAY_STEP,
        LEARNING_RATE_DECAY)
    
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(
                        grad, 
                        -GRADIENT_CLIPPING, 
                        GRADIENT_CLIPPING), 
                  var) 
                  for grad, var in gvs]
    return optimizer.apply_gradients(capped_gvs), global_epoch, learning_rate

def _compile_model(model):
    """Generates the tensor graph for training"""

    print("\nGenerating graph...")
    graph = tf.Graph()
    with graph.as_default():
        # Train input
        tf_train_xs = tf.placeholder(tf.float32, 
                                     shape=(TRAIN_BATCH_SIZE,
                                            IMAGE_HEIGHT, IMAGE_WIDTH,
                                            CHANNELS))
        tf_train_labels = tf.placeholder(tf.int64,
                                         shape=(TRAIN_BATCH_SIZE,
                                                IMAGE_HEIGHT, IMAGE_WIDTH))
        # Validation input
        tf_val_xs = tf.placeholder(tf.float32,
                                   shape=(VAL_BATCH_SIZE,
                                          IMAGE_HEIGHT, IMAGE_WIDTH,
                                          CHANNELS))
        tf_val_labels = tf.placeholder(tf.int64,
                                       shape=(VAL_BATCH_SIZE, 
                                              IMAGE_HEIGHT, IMAGE_WIDTH))

        @memoise_variables("convolutions")
        def make_model(*args, **kwargs):
            return model(*args, **kwargs)

        # Training Logits
        logits = make_model(tf_train_xs, 
                            apply_dropout=True, 
                            dropout_keep_rate=DROPOUT_KEEP_RATE)
        print("Train Logits:", logits)
        
        # L2 loss function
        l2 = L2_LAMBDA * sum([tf.nn.l2_loss(w) 
                         for w in DEFAULT_REGISTRY_MANAGER.get_weight_items()])

        loss, train_prediction = loss_and_predict(logits, tf_train_labels, l2)
        
        # Optimizer
        optimizer, global_epoch, learning_rate = optimize(loss)
        
        # Validation Logits
        val_logits = make_model(tf_val_xs, apply_dropout=False)
        print("Validation logits:", val_logits)
        val_loss, valid_prediction = loss_and_predict(val_logits,
                                                      tf_val_labels,
                                                      l2)
        
        train_f1 = f_beta_score(train_prediction, tf_train_labels)
        val_f1 = f_beta_score(valid_prediction, tf_val_labels)

        param_saver = tf.train.Saver(max_to_keep=CHECK_POINTS_TO_KEEP)
    # # train_writer = tf.train.SummaryWriter('train_sum', graph)
    return locals()

def do_validation(session, X_val, y_val, epoch, **loc):
    """Executes validation routine"""

    val_size = X_val.shape[0]
    num_steps = val_size // VAL_BATCH_SIZE
    val_indices = np.arange(val_size)
    
    f_stats = []
    
    total_count = 0
    total_loss = 0.0
    tp, fp, fn, tn = 0, 0, 0, 0
    
    print('-' * 80)
    for step in range(num_steps):
        print_("\r... performing validation: {}/{}" \
               .format(step, num_steps -1), end='', flush=True)
        
        offset = step * VAL_BATCH_SIZE
        b_indices = val_indices[offset:offset + VAL_BATCH_SIZE]
        batch_x = X_val[b_indices]
        batch_labels = y_val[b_indices]
        
        feed_dict = {
            loc['tf_val_xs']: batch_x,
            loc['tf_val_labels']: batch_labels
        }
        
        _val_loss, _predictions, = session.run(
            [loc['val_loss'], loc['valid_prediction']],
            feed_dict=feed_dict)

        total_count += _predictions.shape[0]
        total_loss += _val_loss
        
        _tp, _fp, _fn, _tn = get_f_stats(_predictions, batch_labels)
        f_stats.append([_tp, _fp, _fn, _tn])
        tp, fp, fn, tn = tp + _tp, fp + _fp, fn + _fn, tn + _tn
        
    save_f_stat_log(F_STATS_VAL_FILE, f_stats, epoch)
    
    val_f1_score = np_f_beta_score(tp, fp, fn)
    print("\nValidation: total_count: {}, F1 score: {}" \
          .format(total_count, val_f1_score))
    print('-' * 80)
    return total_loss / float(num_steps), total_count, val_f1_score

def do_training(session, X_train, y_train,
                X_val=None, y_val=None,
                from_checkpoint=None,
                epochs=1, start_at_epoch=0,
                **loc):
    """Executes the training routine"""

    print('\n' + '=' * 80)
    if from_checkpoint is None:
        tf.initialize_all_variables().run()
        print("Initialized variables.")
    else:
        loc['param_saver'].restore(session, from_checkpoint)
        print("Restored variables from '{}'.".format(from_checkpoint))
        
    train_size = X_train.shape[0]
    train_indices = np.arange(train_size)
    num_steps = train_size // TRAIN_BATCH_SIZE
    print("Iterations per epoch: {}".format(num_steps))
    for epoch in range(start_at_epoch, start_at_epoch + epochs):
        epoch_time = time_log()
        print("\nRunning epoch {} ...".format(epoch))
        loss_log = []
        f_stats = []

        total_count = 0
        total_loss = 0.0
        
        tp, fp, fn, tn = 0, 0, 0, 0  # tp = true positive
                                     # fp = false positive
                                     # fn = false negative
                                     # tn = true negative
        
        np.random.shuffle(train_indices) # Shuffle dataset
        
        for step in range(num_steps):
            b_time = time_log()
            
            offset = step * TRAIN_BATCH_SIZE
            b_indices = train_indices[offset:offset + TRAIN_BATCH_SIZE]
            batch_x = X_train[b_indices]
            batch_labels = y_train[b_indices]
            
            feed_dict = {
                loc['tf_train_xs']: batch_x,
                loc['tf_train_labels']: batch_labels,
                loc['global_epoch']: epoch
            }

            _, _learning_rate, _loss, _predictions, _train_f1 = \
            session.run([loc['optimizer'],
                         loc['learning_rate'],
                         loc['loss'],
                         loc['train_prediction'],
                         loc['train_f1']],
                        feed_dict=feed_dict)

            loss_log.append([_loss, _train_f1])
            total_count += _predictions.shape[0]
            total_loss += _loss
            _tp, _fp, _fn, _tn = get_f_stats(_predictions, batch_labels)
            f_stats.append([_tp, _fp, _fn, _tn])
            tp, fp, fn, tn = tp + _tp, fp + _fp, fn + _fn, tn + _tn
            
            if (step % SHOW_LOG_AT_EVERY_ITERATION == 0):
                print("\n... Epoch {}/{}, iteration {}/{}:" \
                      .format(epoch, start_at_epoch + epochs - 1,
                              step, num_steps - 1))
                print("...     Mini batch loss: {}".format(_loss))
                print("...     F1 score       : {}".format(_train_f1))
                print("...     Elapsed time   : {}".format(time_log() - b_time))
        
        loc['param_saver'].save(session, os.path.join(CHECK_POINT_DIR,
                                               CHECK_POINT_FILE.format(epoch)))
        save_loss_log(loss_log, epoch)
        save_f_stat_log(F_STATS_TRAIN_FILE, f_stats, epoch)
        
        train_f1_score = np_f_beta_score(tp, fp, fn)
        val_tot_loss, val_tot_count, val_f1 = \
                                    (0., 0, 0.) if X_val is None else \
                                    do_validation(session,
                                                  X_val, y_val,
                                                  epoch,
                                                  **loc)

        # Save training statistics
        update_training_log(epoch, step + 1,
                            total_count, val_tot_count,
                            _learning_rate, 
                            total_loss / float(num_steps) , val_tot_loss,
                            train_f1_score, val_f1)

        # Update checkpoint with highest validation score
        save_highest_checkpoint(val_f1, epoch)
        
        print("Epoch {} stats:".format(epoch))
        print("learning rate: {}, average loss: {}" \
              .format(_learning_rate, total_loss / (step + 1)))
        print("training F1: {}, validation F1: {}" \
              .format(train_f1_score, val_f1))
        print("Epoch elapsed time:", time_log() - epoch_time)
        print('-' * 80)
    
    print("Done training at epoch {}.".format(epoch))

def run_training(model, data, 
                 epochs=EPOCHS_TO_RUN, start_at_epoch=START_AT_EPOCH):
    """Initiate training routine"""

    if not(start_at_epoch == 'latest' or \
            (isinstance(start_at_epoch, int) and start_at_epoch >= 0)):
        raise ValueError("Epochs can start at either 'latest' or at >= 0.")

    start_at = start_at_epoch if isinstance(start_at_epoch, int) \
                                 else get_latest_checkpoint_index() + 1
        
    resume_from_checkpoint = None if start_at == 0 \
                                     else get_checkpoint_at_index(start_at - 1)
    
    train_x, train_y, val_x, val_y = data
    print_data_stats("Training Set:", train_x, train_y)
    print_data_stats("Validation Set:", val_x, val_y)

    loc = _compile_model(model)
    with tf.Session(graph=loc['graph']) as sess:
        do_training(sess, train_x, train_y, val_x, val_y,
                    from_checkpoint=resume_from_checkpoint,
                    epochs=epochs, start_at_epoch=start_at,
                    **loc)

# INFERENCE
################################################################################
def get_index_of_highest_checkpoint():
    """Returns the index of the checkpoint with highest score"""
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

def _compile_model_for_classifier(model, batch_size):
    """Generates tensor graph for inference"""

    print("\nGenerating graph...")
    graph = tf.Graph()
    with graph.as_default():
        tf_classif_xs = tf.placeholder(tf.float32,
                                       shape=(batch_size,
                                              IMAGE_HEIGHT, IMAGE_WIDTH,
                                              CHANNELS))

        @memoise_variables("convolutions_classifier")
        def make_model(*args, **kwargs):
            return model(*args, **kwargs)

        # Classifier Logits
        classif_logits = make_model(tf_classif_xs, apply_dropout=False)
        print("Classifier logits:", classif_logits)
        classif_prediction = prediction(classif_logits)

        param_saver = tf.train.Saver(max_to_keep=CHECK_POINTS_TO_KEEP)

    return locals()

def _make_classifier(model, checkpoint_index='latest', batch_size=1):
    """Returns the classifer function"""

    # Get checkpoint file
    if not(checkpoint_index == 'latest' or \
            (isinstance(checkpoint_index, int) and checkpoint_index >= 0)):
        raise ValueError("Checkpoint index can either be 'latest' or >= 0.")

    cp_index = checkpoint_index if isinstance(checkpoint_index, int) \
                                 else get_latest_checkpoint_index()
    checkpoint = get_checkpoint_at_index(cp_index)

    # Compile graph
    loc = _compile_model_for_classifier(model, batch_size)
    session = tf.Session(graph=loc['graph'])

    # Load variables from checkpoint
    loc['param_saver'].restore(session, checkpoint)
    print("Restored variables from '{}'.".format(checkpoint))

    # Normalizer for images
    if os.path.exists(TRAIN_STATS_PICKLE):
        with open(TRAIN_STATS_PICKLE, 'rb') as f:
            stats = cPickle.load(f)
        mean = stats['mean']
        std = stats['std']
        normalize = lambda imgs: (imgs - mean) / std
    else:
        normalize = lambda imgs: (imgs - imgs.mean()) / imgs.std()

    # Core classifier function
    def classifier_func(X):
        count = X.shape[0]
        X = normalize(X)

        # Pad X with empty images if number of images in X
        # does not match with batch size
        if X.shape[0] < batch_size:
            shape = list(X.shape)
            shape[0] = batch_size - shape[0]
            pad = np.zeros(shape, dtype=X.dtype)
            X = np.concatenate((X, pad), axis=0)

        feed_dict = {loc['tf_classif_xs']: X}
        preds = session.run(loc['classif_prediction'],
                                   feed_dict=feed_dict)[:count]
        return np.argmax(preds, axis=3)

    return session, classifier_func
################################################################################

class DeepClassifier:
    """Wrapper class for classifier that uses deep neural network"""

    def __init__(self, model):
        """Creates a new instance of DeepClassifier

        Arguments:
            model - A function that accept data input returns a tensor
        """

        self.model = model
        self.checkpoint_index = None
        self._session = None
        self._classifier = None

    def fit(self, X, y, X_val=None, y_val=None):
        """Executes training on the dataset

        Arguments:
            X     - Images for training
            y     - Mask label of training images
            X_val - Images for validation
            y_val - Mask labels of validation images 

        If X_val is none, no validation will be executed.
        """

        data = (X, y, X_val, y_val)
        run_training(model=self.model, data=data)
        self.load_from_checkpoint(index='highest')

    def predict(self, X, batch_size=1, regen=False):
        """Returns prediction labels of X

        Arguments:
            X          - Images to be predicted
            batch_size - Mini-batch size to use during inference
            regen      - If true, force the tensor graph to be regenerated
        """

        self._init_for_inference(batch_size=batch_size, regen=regen)

        batch_count = int(ceil(float(X.shape[0]) / batch_size))
        x_iter = lambda: (X[i * batch_size : i * batch_size + batch_size] \
                          for i in range(batch_count))

        preds = [self._classifier(x) for x in x_iter()]
        return preds[0] if len(preds) == 1 else np.concatenate(preds, axis=0)

    def load_from_checkpoint(self, index='highest',
                                   init_for_inference=False,
                                   batch_size=1):
        """Sets the checkpoint index to be used for inference

        Arguments:
            index              - Index number of the checkpoint file. Specify
                                 'highest' to use the checkpoint that has the
                                 highest score. Specify 'latest' to use the
                                 latest saved checkpoint file. Set to an integer
                                 value to use that specificcheckpoint file.
            init_for_inference - If true, generates generates tensor graph for
                                 inference and iniate it with the checkpoint
                                 file.
            batch_size         - Mini-batch size to use during inference
        """

        self.checkpoint_index = get_index_of_highest_checkpoint() \
                                    if index == 'highest' \
                                    else index
        if init_for_inference:
            self._init_for_inference(batch_size=batch_size, regen=True)

    def _init_for_inference(self, batch_size=1, regen=False):
        """Generates tensor graph for inference and initialize the variables
        for the checkpoin file.

        Arguments:
            batch_size - Mini-batch size to use during inference
            regen      - For the tensor graph to be regenerated
        """

        if self.checkpoint_index is None:
            raise RuntimeError("Model is not yet built.")

        if regen: self._delete_session()
        if self._session is None:
            session, classifier = _make_classifier(self.model,
                                                self.checkpoint_index,
                                                batch_size)
            self._set_session(session)
            self._classifier = classifier

    def _set_session(self, session):
        self._delete_session()
        self._session = session

    def _delete_session(self):
        if self._session is not None:
            self._session.close()
            self._session = None
            self._classifier = None

    def __del__(self):
        """Make sure session is closed when object is destroyed"""
        self._delete_session()

