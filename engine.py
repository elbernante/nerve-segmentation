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
    if _class_weights['positive'] is None or \
            _class_weights['negative'] is None:
       pos_class_weight, neg_class_weight = get_class_weights()
       _class_weights['positive'] = pos_class_weight
       _class_weights['negative'] = neg_class_weight

    return tf.constant([ _class_weights['negative'],
                         _class_weights['positive']])

def get_train_val_sets(image_dir=TRAIN_DIR):
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
    combed_logits = comber(logits)
    weighted_logits = tf.mul(combed_logits, class_weights())
    cross_en = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        weighted_logits, tf.reshape(targets, [-1]))
    loss = tf.reduce_mean(cross_en) + l2
    pred = uncomber(tf.nn.softmax(combed_logits))
    return loss, pred

def prediction(logits):
    return uncomber(tf.nn.softmax(comber(logits)))

def optimize(loss):
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

def _compile_model_for_inference(model):
    print("\nGenerating graph...")
    graph = tf.Graph()
    with graph.as_default():
        tf_test_xs = tf.placeholder(tf.float32,
                                    shape=(TEST_BATCH_SIZE,
                                           IMAGE_HEIGHT, IMAGE_WIDTH,
                                           CHANNELS))

        @memoise_variables("convolutions")
        def make_model(*args, **kwargs):
            return model(*args, **kwargs)

        # Test Logits
        test_logits = make_model(tf_test_xs, apply_dropout=False)
        print("Test logits:", test_logits)
        test_prediction = prediction(test_logits)

        param_saver = tf.train.Saver(max_to_keep=CHECK_POINTS_TO_KEEP)

    return locals()

def do_validation(session, X_val, y_val, epoch, **loc):
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


# TODO: Remove block
################################################################################
import cv2
from lib.file_io import create_if_not_exists

def get_img_0(img, lbl):
    img = (img.squeeze() - img.min())/(img.max() - img.min())
   
    h = ((img * 0.9) + 0.1) * lbl.astype(np.float32)
    d = img * 0.4 * np.logical_not(lbl).astype(np.float32)
    overlay = d + h

    cvt = lambda i: cv2.cvtColor((i * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return [cvt(i) for i in [img, overlay]], img, lbl.astype(np.float32)

create_if_not_exists('output/preds/')
def save_pred(index, b_indices, _predictions, epoch, step, src_img=None):
    sel_pred = np.where(b_indices == index, True, False)
    
    
    jet = lambda i: cv2.applyColorMap(i, cv2.COLORMAP_JET)
    hot = lambda i: cv2.applyColorMap(i, cv2.COLORMAP_HOT)
    gray = lambda i: cv2.cvtColor((i * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    soft = lambda i: (((i - min(0, i.min())) / (max(1, i.max()) - min(0, i.min()))) * 255).astype(np.uint8)
    scale = lambda i: (((i - i.min()) / (i.max() - i.min())) * 255).astype(np.uint8)
    

    pos = _predictions[sel_pred][0][:,:,1]
    neg = _predictions[sel_pred][0][:,:,0]
    pred = np.argmax(_predictions[sel_pred][0], axis=2)
    
    diff = np.zeros_like(pred, dtype=np.uint8) if src_img is None else \
           ((src_img[2] * 2.) + pred.astype(np.float32)) / 3.0
        
    preds = np.concatenate((jet(soft(neg)), jet(soft(pos)), gray(pred)), axis=1)
    heats = np.concatenate((jet(scale(neg)), jet(scale(pos)), hot(soft(diff))), axis=1)
    
    if src_img is not None:
        h = ((src_img[1] * 0.9) + 0.1) * pred.astype(np.float32)
        d = src_img[1] * 0.4 * np.logical_not(pred).astype(np.float32)
        overlay = d + h
    
        src = np.concatenate([src_img[0][0], src_img[0][1], gray(overlay)], axis=1)
        imgs = np.concatenate((preds, heats, src), axis=0)
    else:
        imgs = np.concatenate((preds, heats), axis=0)
        
    cv2.imwrite('output/preds/{}-{}-{}.tif'.format(index, epoch, step), imgs)
################################################################################


def do_training(session, X_train, y_train,
                X_val=None, y_val=None,
                from_checkpoint=None,
                epochs=1, start_at_epoch=0,
                **loc):

    print('\n' + '=' * 80)
    if from_checkpoint is None:
        tf.initialize_all_variables().run()
        print("Initialized variables.")
    else:
        loc['param_saver'].restore(session, from_checkpoint)
        print("Restored variables from '{}'.".format(from_checkpoint))

    # TODO: Delete this line
    img_0 = get_img_0(X_train[0], y_train[0])
        
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
        
        np.random.shuffle(train_indices)
        
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
            
            
            # print("Batch elapsed:", time_log() - b_time)
            
            # TODO: Delete this block
            # ==========================================
            if 0 in b_indices:
                save_pred(0, b_indices, _predictions,
                          epoch, step, src_img=img_0)
            # ==========================================
            
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

        update_training_log(epoch, step + 1,
                            total_count, val_tot_count,
                            _learning_rate, 
                            total_loss / float(num_steps) , val_tot_loss,
                            train_f1_score, val_f1)

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
        # do_training(sess, train_x[:8], train_y[:8], val_x[:8], val_y[:8],
        do_training(sess, train_x, train_y, val_x, val_y,
#         do_training(sess, X_train, y_train, None, None,
                    from_checkpoint=resume_from_checkpoint,
                    epochs=epochs, start_at_epoch=start_at,
                    **loc)

# INFERENCE
################################################################################
def get_test_image_set(image_dir=TEST_DIR):
    prefix_len = len(os.path.join(image_dir, ''))
    imgs = glob.glob(os.path.join(image_dir, "*[0-9].{}".format(FILE_EXT)))
    to_int = lambda k: int(k)
    return sorted([s[prefix_len:s.rindex('.')] for s in imgs], key=to_int)

def get_test_batch_data(img_keys, image_dir=TEST_DIR):
    f_names = [os.path.join(image_dir, "{}.{}".format(img_key, FILE_EXT))
               for img_key in img_keys]
    imgs = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in f_names],
                    dtype=np.float32) \
             .reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    return imgs

from itertools import tee, chain
def run_length(label, neg_class=0, 
               enumerate=enumerate, next=next, zip=zip, 
               tee=tee, chain=chain, np=np):
    """Returns generator of run length encoding as (position, length) pair of tuples.
    
    Parameters:
        label      - Required. numpy array with shape (H, W).
                     Expected to have binary values.
        neg_class  - Optional. Value to be classified as the negative class.
                     All other values are classified as positive class.
                     Defaults to 0.
    
    All other parameters are not strictly necessary but are declared as default values
    to avoid global lookups and thereby have performance gains.
    
    Returns:
        A generator for a list of tuples. Each tuple is a (position, length) pair.
    """
    
    padded = chain([neg_class], np.ravel(label, order='F'), [neg_class])
    a, b = tee(padded); next(b, None)
    switches = (i + 1 for i, (a, b) in enumerate(zip(a, b)) if a != b)
    return ((p, next(switches) - p) for p in switches)

# Utility to concatenate run length encoded tuples to string
string_rle = lambda rle: ' '.join(['{} {}'.format(p, l) for p, l in rle])

def to_run_length(batch_keys, preds):
    def process_pred(k, p):
        discreet = np.argmax(p, axis=2)
        rle = string_rle(run_length(discreet))
        return [k, rle]
    
    return [process_pred(*p) for p in zip(batch_keys, preds)]

from lib.file_io import write_to_csv
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")
def do_inference(session, test_keys, checkpoint, image_dir=TEST_DIR, **loc):
    inference_time = time_log()
    print('\n' + '=' * 80)
    loc['param_saver'].restore(session, checkpoint)
    print("Restored variables from '{}'.".format(checkpoint))
    
    if len(test_keys) % TEST_BATCH_SIZE != 0:
        print("WARNING: Number of items is not divisible by the batch size." +
              " Some items will not be processed.")

    # Normalizer for images
    with open(TRAIN_STATS_PICKLE, 'rb') as f:
        stats = cPickle.load(f)
    mean = stats['mean']
    std = stats['std']
    normalize = lambda imgs: (imgs - mean) / std
    
    test_size = len(test_keys)
    num_steps = test_size // TEST_BATCH_SIZE
    rles = []
    
    eta = 'estimating...'
    for step in range(num_steps):
        b_time = time_log()
        print_("\r... running inference: {}/{} | ETA: {}" \
                    .format(step, num_steps - 1, eta),
               end='', flush=True)
        
        offset = step * TEST_BATCH_SIZE
        batch_keys = test_keys[offset:offset + TEST_BATCH_SIZE]
        batch_x = normalize(get_test_batch_data(batch_keys, 
                                                image_dir=image_dir))
        
        feed_dict = {loc['tf_test_xs']: batch_x}
        
        _predictions = session.run(loc['test_prediction'],
                                   feed_dict=feed_dict)
        
        rles.extend(to_run_length(batch_keys, _predictions))
        
        b_elapsed = time_log() - b_time
        eta_sec = (num_steps - step) * b_elapsed
        eta = "{:.3f} seconds".format(eta_sec) if eta_sec < 60.0 else \
              "{:.3f} minutes".format(eta_sec / 60.0)
    
    # Create sumission file
    write_to_csv(SUBMISSION_FILE, ['img', 'pixels'], rles)
    
    print("\nDone processing {} images. Elapsed time: {}.".format(
            len(test_keys), time_log() - inference_time))

def run_inference(model, image_dir=TEST_DIR, checkpoint_index='latest'):
    if not(checkpoint_index == 'latest' or \
            (isinstance(checkpoint_index, int) and checkpoint_index >= 0)):
        raise ValueError("Checkpoint index can either be 'latest' or >= 0.")
    
    cp_index = checkpoint_index if isinstance(checkpoint_index, int) \
                                 else get_latest_checkpoint_index()        
    checkpoint = get_checkpoint_at_index(cp_index)
    test_keys = get_test_image_set(image_dir=image_dir)
    
    loc = _compile_model_for_inference(model)
    with tf.Session(graph=loc['graph']) as sess:
        do_inference(sess, test_keys, checkpoint, image_dir=image_dir, **loc)
        # do_inference(sess, test_keys[:8], checkpoint, image_dir=image_dir, **loc)

################################################################################

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

def _compile_model_for_classifier(model, batch_size):
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

class DeepClassifier:
    def __init__(self, model):
        self.model = model
        self.checkpoint_index = None
        self._session = None
        self._classifier = None

    def fit(self, X, y, X_val=None, y_val=None):
        data = (X, y, X_val, y_val)
        run_training(model=self.model, data=data)
        self.load_from_checkpoint(index='highest')

    def predict(self, X, batch_size=1, regen=False):
        self._init_for_inference(batch_size=batch_size, regen=regen)

        batch_count = int(ceil(float(X.shape[0]) / batch_size))
        x_iter = lambda: (X[i * batch_size : i * batch_size + batch_size] \
                          for i in range(batch_count))

        preds = [self._classifier(x) for x in x_iter()]
        return preds[0] if len(preds) == 1 else np.concatenate(preds, axis=0)

    def load_from_checkpoint(self, index='highest',
                                   init_for_inference=False,
                                   batch_size=1):
        self.checkpoint_index = get_index_of_highest_checkpoint() \
                                    if index == 'highest' \
                                    else index
        if init_for_inference:
            self._init_for_inference(batch_size=batch_size, regen=True)

    def _init_for_inference(self, batch_size=1, regen=False):
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
        self._delete_session()

