from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os, glob
import argparse

import numpy as np
import cv2
from tqdm import tqdm       # Progress bar

from engine import DeepClassifier
from train import make_model
from lib.file_io import create_if_not_exists
from config import IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS

IMAGE_FILE_EXT = 'tif'
MASK_SUFFIX = '_mask'

HEIGHT_WIDTH_RATIO = 420.0/580.0

def get_args():
    parser = argparse.ArgumentParser(
                 description='Search for brachial plexus in ultrasound images.')
    parser.add_argument('-s', '--source-dir', dest='src_dir',
                        type=str, required=True,
                        help='Source directory for images.')
    parser.add_argument('-d', '--destination-dir', dest='dest_dir',
                        type=str, default=None,
                        help='Destination directory for prediction output.' +
                             ' Defaults to <SRC_DIR>_output.')
    parser.add_argument('-m', '--mask-only',
                        action='store_true',
                        help='Output masks only. No overlay on image.')

    args = vars(parser.parse_args())
    if args['dest_dir'] is None:
        args['dest_dir'] = args['src_dir'] + '_output'
    return args

def get_target_keys(src_dir):
    imgs = glob.glob(os.path.join(src_dir, '*.{}'.format(IMAGE_FILE_EXT)))
    fnames = [os.path.basename(f) for f in imgs]
    keys = [f.rsplit('.', 1)[0] for f in fnames]
    keys = [f.rsplit(MASK_SUFFIX, 1)[0] for f in keys]
    return list(set(keys))

def get_image(src_dir, key):
    f = os.path.join(src_dir, '{}.{}'.format(key, IMAGE_FILE_EXT))
    assert os.path.isfile(f), "Image file '{}' does not exists!".format(f)
    return cv2.imread(f, cv2.IMREAD_GRAYSCALE)

def get_image_label(src_dir, key):
    f = os.path.join(src_dir, 
                     '{}{}.{}'.format(key, MASK_SUFFIX, IMAGE_FILE_EXT))
    if not(os.path.isfile(f)): return None
    return cv2.imread(f, cv2.IMREAD_GRAYSCALE) // 255

def crop_and_resize(img):
    if (IMAGE_HEIGHT, IMAGE_WIDTH) == img.shape:
        return img, img.shape

    # Center crop to proportion
    ratio = float(img.shape[0]) / img.shape[1]
    if ratio != HEIGHT_WIDTH_RATIO:
        if ratio > HEIGHT_WIDTH_RATIO:
            w = img.shape[1]
            h = int(w * HEIGHT_WIDTH_RATIO)
        else:
            h = img.shape[0]
            w = int(h / HEIGHT_WIDTH_RATIO)

        h0 = (img.shape[0] - h) // 2
        h1 = h0 + h
        w0 = (img.shape[1] - w) // 2
        w1 = w0 + w
        img = img[h0:h1, w0:w1]
    
    crop_size = tuple(img.shape)

    # resize to target dimension
    interpolation = cv2.INTER_AREA \
                        if img.size > IMAGE_WIDTH * IMAGE_HEIGHT else \
                        cv2.INTER_CUBIC
    img = cv2.resize(img,
                     (IMAGE_WIDTH, IMAGE_HEIGHT), 
                      interpolation=interpolation)
    return img, crop_size

def smooth_resize_to(img, height, width):
    if (height, width) == img.shape: return img
    if img.size < height * width:
        img_o = cv2.resize(img,
                           (width, height), 
                           interpolation=cv2.INTER_CUBIC)
        img_o = cv2.GaussianBlur(img_o, (31, 31),  0)
    else:
        img_o = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    return cv2.threshold(img_o, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def zero_pad(img, height, width):
    if (height, width) == img.shape: return img
    if not(img.shape[0] < height or img.shape[1] < width): return img
    padded = np.zeros((height, width))
    h = (height - img.shape[0]) // 2
    w = (width - img.shape[1]) // 2
    padded[h:h+img.shape[0], w:w+img.shape[1]] = img
    return padded

def overlay_prediction(img, pred, mask=None, outline_size=1):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Highlight mask
    if mask is not None:
        img[:,:,2] *= np.logical_not(mask)
        edges = cv2.Canny(mask * 255, 100, 200) > 0
        img[edges, 0]= 255
        img[edges, 1]= 255
        img[edges, 2]= 0


    # Outline prediction
    edges = cv2.Canny(pred, 100, 200)
    if outline_size == 1:
        edges = edges > 0
    else:
        kernel = np.ones((outline_size, outline_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations = 1) > 0
    img[edges, 0]= 0
    img[edges, 1]= 255
    img[edges, 2]= 255

    return img

def run_inference():
    args = get_args()
    keys = get_target_keys(args['src_dir'])
    if len(keys) == 0:
        print("No images found. Expected filename extenion: {}" \
             .format(IMAGE_FILE_EXT))
        exit(0)

    create_if_not_exists(args['dest_dir'])

    cls = DeepClassifier(make_model)
    cls.load_from_checkpoint('highest', init_for_inference=True)

    for k in tqdm(keys, desc='Processing', unit='img'):
        raw_img = get_image(args['src_dir'], k)
        img, crop_size = crop_and_resize(raw_img)
        img = img.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
        
        pred = cls.predict(img)
        
        resized = smooth_resize_to(pred[0].astype(np.uint8) * 255, *crop_size)
        padded = zero_pad(resized, *raw_img.shape)

        if args['mask_only']:
            output = padded
        else:
            mask  = get_image_label(args['src_dir'], k)
            output = overlay_prediction(raw_img, padded, mask, outline_size=2)

        cv2.imwrite(os.path.join(args['dest_dir'], '{}.png'.format(k)),
                        output)
    
    print("Done. {} images saved to '{}'".format(len(keys), args['dest_dir']))

if __name__ == '__main__':
    run_inference()
