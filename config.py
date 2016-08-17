import os
from lib.file_io import create_if_not_exists as mkdir

# Hyper-parameters
LEARNING_RATE = 1e-5
LEARNING_RATE_DECAY = 0.98
DECAY_STEP = 1
DROPOUT_KEEP_RATE = 0.5
GRADIENT_CLIPPING = 2.0
L2_LAMBDA = 5e-3

# Data information
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 128
CHANNELS = 1

PIXEL_DEPTH = 255

LABELS = [0, 1]
NUM_LABELS = len(LABELS)

# Run configurations
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4

EPOCHS_TO_RUN = 500
START_AT_EPOCH = 'latest'
CHECK_POINTS_TO_KEEP = 20

SHOW_LOG_AT_EVERY_ITERATION = 1

SIMILARTY_THRESHOLD = 0.7       # For filtering out similar images
WITHHOLD_FOR_VALIDATION = 0.2

# Directory configurations
DATA_DIR = mkdir("data")
OUTPUT_DIR = mkdir('output')

TRAIN_SRC = os.path.join(DATA_DIR, "train")
TRAIN_DIR = os.path.join(DATA_DIR, "train_xs96f")
TEST_DIR = os.path.join(DATA_DIR, "test_xs96")
FILE_EXT = "tif"

TRAIN_SET_PICKLE = os.path.join(DATA_DIR, "train_set.npz")
VALIDATION_SET_PICKLE = os.path.join(DATA_DIR, "validation_set.npz")
TRAIN_STATS_PICKLE = os.path.join(DATA_DIR, "train_stats.pkl")

CHECK_POINT_DIR = mkdir(os.path.join(OUTPUT_DIR, "checkpoints"))
CHECK_POINT_FILE = "model.ckpt-{}"    # model.ckpt-{epoch}

HIGHEST_DIR = mkdir(os.path.join(OUTPUT_DIR, "highest_score"))
HIGHEST_SCORE_FILE = "highest_score.pkl"

TRAINING_LOG = os.path.join(OUTPUT_DIR, "training_log.csv")

LOSS_LOG_DIR = mkdir(os.path.join(OUTPUT_DIR, "loss_log"))
LOSS_LOG_FILE = "loss-{}.csv"         # loss-{epoch}.csv

F_STATS_LOG_DIR = mkdir(os.path.join(OUTPUT_DIR, "f_stats_log"))
F_STATS_FILE = "f_stats-{}.csv"      # f_stats-{epoch}.csv

F_STATS_TRAIN_DIR = mkdir(os.path.join(F_STATS_LOG_DIR, "training"))
F_STATS_TRAIN_FILE = os.path.join(F_STATS_TRAIN_DIR, F_STATS_FILE)

F_STATS_VAL_DIR = mkdir(os.path.join(F_STATS_LOG_DIR, "validation"))
F_STATS_VAL_FILE = os.path.join(F_STATS_VAL_DIR, F_STATS_FILE)
