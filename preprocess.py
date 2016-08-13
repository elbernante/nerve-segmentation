from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os, glob

import cv2
import numpy as np
from tqdm import tqdm       # Progress bar

from six import  print_
from six.moves import range, zip, cPickle

from lib.file_io import create_if_not_exists, del_files
from config import *

def get_patient_ids(image_dir=TRAIN_SRC):
    """Returns list of patient IDs"""
    prefix_len = len(os.path.join(image_dir, ''))
    imgs = glob.glob(os.path.join(image_dir, 
                     "*[0-9]_*[0-9].{}".format(FILE_EXT)))
    return sorted(set([int(s[prefix_len:].split('_')[0]) for s in imgs]))


def get_image_ids(image_dir=TRAIN_SRC):
    """Returns a list of all image IDs in a directory"""

    prefix_len = len(os.path.join(image_dir, ''))
    imgs = glob.glob(os.path.join(image_dir, 
                     "*[0-9]_*[0-9].{}".format(FILE_EXT)))
    to_int = lambda k: tuple([int(i) for i in k.split('_')])
    return sorted([s[prefix_len:s.rindex('.')] for s in imgs], key=to_int)


def get_images_for_patient(patient_id, image_dir=TRAIN_SRC):
    """Returns images IDs of a patient"""

    prefix_len = len(os.path.join(image_dir, ''))
    imgs = glob.glob(os.path.join(image_dir, 
                     "{}_*[0-9].{}".format(patient_id, FILE_EXT)))
    to_int = lambda k: tuple([int(i) for i in k.split('_')])
    return sorted([s[prefix_len:s.rindex('.')] for s in imgs], key=to_int)


def get_images_for_keys(img_keys, image_dir=TRAIN_SRC):
    f_names = [os.path.join(image_dir, "{}.{}".format(img_key, FILE_EXT))
                for img_key in img_keys]
    return np.asarray([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in f_names],
                dtype=np.float32)


def get_image_labels_for_keys(img_keys, image_dir=TRAIN_SRC):
    """Returns the label of images"""

    f_names = [os.path.join(image_dir, "{}_mask.{}".format(img_key, FILE_EXT))
               for img_key in img_keys]
    return np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in f_names], 
                    dtype=np.int64) // int(PIXEL_DEPTH)


def split_by_label(img_ids, image_dir=TRAIN_SRC):
    """Segregates images whether or not it has label annotation"""

    has_labels = get_image_labels_for_keys(img_ids, image_dir=image_dir) \
                 .sum(axis=(1,2)) > 0
    with_labels = [i for l, i in zip(has_labels, img_ids) if l]
    no_labels = [i for l, i in zip(has_labels, img_ids) if not l]
    return (with_labels, no_labels)


def stratified_train_val_split(labels, train=0.7, seed=None):
    """Stratify split data into training-validation sets"""

    saved_state = np.random.get_state()
    np.random.seed(seed)

    y = np.asarray(labels)
    train_idx = np.zeros(y.shape[0], dtype=np.bool)
    val_idx = np.zeros(y.shape[0], dtype=np.bool)
    classes = np.unique(y)
    for cls in classes:
        idx = np.nonzero(y == cls)[0]
        np.random.shuffle(idx)
        n = int(train * idx.shape[0])

        train_idx[idx[:n]] = True
        val_idx[idx[n:]] = True
    
    t_idx = np.nonzero(train_idx)[0]
    np.random.shuffle(t_idx)
    v_idx = np.nonzero(val_idx)[0]
    np.random.shuffle(v_idx)
    
    np.random.set_state(saved_state)
    return t_idx, v_idx

def group_stratified_train_val_split(labels, img_ids, train=0.7, seed=None):
    """Stratify split data into training-validation sets based on
    group ID and presence of mask of the data
    """

    g_label = lambda img_id, lbl: '{}_{}'.format(img_id.split('_')[0], lbl)
    zip_g_lbl = np.vectorize(g_label, otypes=[np.str])
    grouped_labels = zip_g_lbl(img_ids, labels)
    return stratified_train_val_split(grouped_labels, train=train, seed=seed)


def get_mask_patch(img_key, image_dir=TRAIN_SRC):
    """Returns the rectangle area of the image where there is annotation"""

    mask = cv2.imread(os.path.join(image_dir, 
                      "{}_mask.{}".format(img_key, FILE_EXT)), 
                      cv2.IMREAD_GRAYSCALE)
    pos = np.where(mask > 127)
    min_y, max_y = min(pos[0]), max(pos[0]) + 1
    min_x, max_x = min(pos[1]), max(pos[1]) + 1

    img = cv2.imread(os.path.join(image_dir, 
                     "{}.{}".format(img_key, FILE_EXT)), 
                     cv2.IMREAD_GRAYSCALE)
    
    return (img[min_y:max_y, min_x:max_x], (min_y, min_x))


def get_similarity(img_key, patch, loc_y, loc_x, image_dir=TRAIN_SRC):
    """Returns the similarity value of a given patch in an image"""

    img = cv2.imread(os.path.join(image_dir, 
                     "{}.{}".format(img_key, FILE_EXT)), 
                     cv2.IMREAD_GRAYSCALE)
    offset_h, offset_w = int(img.shape[0] // 8), int(img.shape[1] // 8)
    min_y = max(0, loc_y - offset_h)
    max_y = min(img.shape[0], loc_y + patch.shape[0] + offset_h)
    
    min_x = max(0, loc_x - offset_w)
    max_x = min(img.shape[1], loc_x + patch.shape[1] + offset_w)
    
    scan_area = img[min_y:max_y, min_x:max_x]
    
    scanned = cv2.matchTemplate(scan_area, patch, cv2.TM_CCOEFF_NORMED)
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(scanned)
    return (maxVal, (maxLoc[1] + min_y, maxLoc[0] + min_x))


def clean_up_no_labels(with_labels, no_labels, 
                       threshold=SIMILARTY_THRESHOLD, 
                       image_dir=TRAIN_SRC):
    """Filters out images that have no labels but are very similar to
    images that have label.
    """

    pbar = tqdm(total=len(with_labels), 
                desc='   Filtering images', 
                leave=False)
    for wl in with_labels:
        patch, loc = get_mask_patch(wl, image_dir=image_dir)
        sim, _ = zip(*[get_similarity(nl, patch, *loc) for nl in no_labels])
        no_labels = [nl for nl, s in zip(no_labels, sim ) if s < threshold]
        if len(no_labels) == 0: break
        pbar.update()
    pbar.close()
        
    return no_labels


def generate_src_dst_for_keys(img_keys, 
                              src_folder=TRAIN_SRC, 
                              dst_folder=TRAIN_DIR):
    """Returns list of source-destination filename pairs"""

    imgs = ('{}.{}'.format(k, FILE_EXT) for k in img_keys)
    masks = ('{}_mask.{}'.format(k, FILE_EXT) for k in img_keys)
    f_names = [f for im in zip(imgs, masks) for f in im]

    src = [os.path.join(src_folder, f) for f in f_names]
    dest = [os.path.join(dst_folder, f) for f in f_names]
    return src, dest


def resize_img(src_file, dst_file):
    """Resizes src_file image and save it to dst_file"""

    img = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img,
                         (IMAGE_WIDTH, IMAGE_HEIGHT), 
                         interpolation=cv2.INTER_AREA)
    
    if src_file.endswith("_mask.{}".format(FILE_EXT)):
        resized[resized < 127] = 0
        resized[resized >= 127] = 255
    
    cv2.imwrite(dst_file, resized)


def process_patient(patient_id, src_dir=TRAIN_SRC, dst_dir=TRAIN_DIR):
    """ Filters similar images and resizes them"""

    # Filter similar images
    imgs = get_images_for_patient(patient_id, image_dir=src_dir)
    with_labels, no_labels = split_by_label(imgs, image_dir=src_dir)
    no_labels = clean_up_no_labels(with_labels, no_labels,
                                   threshold=SIMILARTY_THRESHOLD,
                                   image_dir=src_dir)

    # Generate source - destanation filenames for resizing
    src, dest = generate_src_dst_for_keys(with_labels + no_labels,
                              src_folder=src_dir,
                              dst_folder=dst_dir)

    # Make sure destation directory exists
    create_if_not_exists(dst_dir)

    # Resize images
    s_d = zip(src, dest)
    for s, d in tqdm(s_d,
                     desc='    Resizing images', 
                     total=len(src),
                     leave=False): 
        resize_img(s, d)


def filter_and_resize(src_dir=TRAIN_SRC, dst_dir=TRAIN_DIR):
    """Filters and resizes images"""

    # Make sure destination directory is clean
    del_files(os.path.join(TRAIN_DIR, '*'))

    print_("Reading raw dataset...", end='', flush=True)
    patient_ids = get_patient_ids(image_dir=src_dir)
    print(" Done - {} patients found".format(len(patient_ids)))

    print("Filtering and resizing images...")

    for pid in tqdm(patient_ids, desc='Processing patients'):
        process_patient(pid, src_dir=src_dir, dst_dir=dst_dir)

    print("Done. Filtered and resized images are saved in: '{}'" \
          .format(dst_dir))


def split_data_train_val(image_dir=TRAIN_DIR):
    """Splits data into training-validation set"""

    # Read images in the directory
    print_("Reading images for training...", end='', flush=True)
    img_keys = np.asarray(get_image_ids(image_dir=image_dir))
    labels = (get_image_labels_for_keys(img_keys, 
                                        image_dir=image_dir) \
             .sum(axis=(1, 2)) > 0).astype(np.int64)
    print(' Done - {} images found'.format(labels.shape[0]))

    # Split data into training-validation sets
    t_idx, v_idx = group_stratified_train_val_split(labels, img_keys,
                                        train=(1.0 - WITHHOLD_FOR_VALIDATION), 
                                        seed=10)

    t_keys = img_keys[t_idx]    # training images keys
    v_keys = img_keys[v_idx]    # validation images keys

    # Training set
    print_("Saving training dataset...", end='', flush=True)
    t_x = get_images_for_keys(t_keys, image_dir=image_dir) \
              .reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    
    t_mean = t_x.mean()     # train set average
    t_std = t_x.std()       # train set standard deviation
    
    t_x -= t_mean           # Center
    t_x /= t_std            # Normalize

    # Training masks
    t_y = get_image_labels_for_keys(t_keys, image_dir=image_dir)

    # Save training set to file
    np.savez(TRAIN_SET_PICKLE, x=t_x, y=t_y)
    print_(" Done - File saved to '{}'".format(TRAIN_SET_PICKLE), flush=True)


    # Validation set
    print_("Saving validation dataset...", end='', flush=True)
    v_x = get_images_for_keys(v_keys, image_dir=image_dir) \
              .reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    v_x -= t_mean           # Center
    v_x /= t_std            # Normalize
    
    # Validation masks
    v_y = get_image_labels_for_keys(v_keys, image_dir=image_dir)
    
    # Save validation set to file
    np.savez(VALIDATION_SET_PICKLE, x=v_x, y=v_y)
    print_(" Done - File saved to '{}'".format(VALIDATION_SET_PICKLE), 
           flush=True)

    # Save training stats for easy loading in future use
    train_tolal_px = t_y.size
    train_pos_count = t_y.sum()
    train_neg_count = train_tolal_px - train_pos_count

    with_labels = t_y[t_y.sum(axis=(1,2)) > 0]
    pos_freq = float(with_labels.sum()) / with_labels.size
    neg_freq = float(train_neg_count) / train_tolal_px

    with open(TRAIN_STATS_PICKLE, 'wb') as f:
        cPickle.dump({'mean': float(t_mean),
                      'std': float(t_std),
                      'total_px': int(train_tolal_px),
                      'pos_count': int(train_pos_count),
                      'neg_count': int(train_neg_count),
                      'img_with_labels:': int(with_labels.shape[0]),
                      'pos_freq': float(pos_freq),
                      'neg_freq': float(neg_freq)},
                      f)

    print("Train data statistics: ")
    print("                        Mean: {}".format(t_mean))
    print("          Standard deviation: {}".format(t_std))
    print("                Total pixels: {}".format(train_tolal_px))
    print("              Positive class: {}".format(train_pos_count))
    print("              Negative class: {}".format(train_neg_count))
    print("           Images with masks: {}".format(with_labels.shape[0]))
    print("    Positive class frequency: {}".format(pos_freq))
    print("    Negative class frequency: {}".format(neg_freq))


# Filter and resize images
filter_and_resize(src_dir=TRAIN_SRC, dst_dir=TRAIN_DIR) 

# Split data into training-validation set
split_data_train_val(image_dir=TRAIN_DIR)
