from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from math import ceil
from lib.tensorflow_wrapper import TensorOp

from six.moves import range, zip

from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_LABELS
from config import TRAIN_DIR
from engine import get_train_val_sets, run_training

def make_model(data, apply_dropout=False, dropout_keep_rate=0.8):
    def conv_image(root_img):
        return root_img.conv(7, 32, stride=1, pad='SAME') \
                       .conv(7, 32, stride=1, pad='SAME') \
                       .conv(3, 64, stride=1, pad='SAME') \
                       .conv(3, 64, stride=1, pad='SAME') \
                       .conv(3, 128, stride=1, pad='SAME') \
                       .conv(3, 256, stride=1, pad='SAME')
    root = TensorOp(data)
    
    DE_ = 32    # depth for the last layer before concatination
    
    # full_layer
    full_layer  = conv_image(root).conv(1, DE_) \
                  .dropout(dropout_keep_rate, apply_dropout)
    
    # 1/2 size
    size_1_2 = root.resize_to(scale=0.5)
    layer_1_2 = conv_image(size_1_2) \
                .incep_a().incep_a() \
                .resize_to(IMAGE_HEIGHT, IMAGE_WIDTH) \
                .conv(1, DE_).conv(3, DE_).conv(3, DE_) \
                .dropout(dropout_keep_rate, apply_dropout)
    
    # 1/4 size
    size_1_4 = size_1_2.resize_to(scale=0.5)
    layer_1_4 = conv_image(size_1_4) \
                .incep_b(dt1=(128, 128, 192),
                         dt2=(128, 128, 128, 128, 192)) \
                .incep_b().incep_b() \
                .resize_to(ceil(IMAGE_HEIGHT / 2), ceil(IMAGE_WIDTH / 2)) \
                .conv(1, DE_ * 2).conv(3, DE_ * 2).conv(3, DE_ * 2) \
                .resize_to(IMAGE_HEIGHT, IMAGE_WIDTH) \
                .conv(1, DE_).conv(3, DE_).conv(3, DE_) \
                .dropout(dropout_keep_rate, apply_dropout)
    
    # 1/8 size
    size_1_8 = size_1_4.resize_to(scale=0.5)
    layer_1_8 = conv_image(size_1_8) \
                .incep_c().incep_c() \
                .resize_to(ceil(IMAGE_HEIGHT / 4), ceil(IMAGE_WIDTH / 4)) \
                .conv(1, DE_ * 4).conv(3, DE_ * 4).conv(3, DE_ * 4) \
                .resize_to(ceil(IMAGE_HEIGHT / 2), ceil(IMAGE_WIDTH / 2)) \
                .conv(1, DE_ * 2).conv(3, DE_ * 2).conv(3, DE_ * 2) \
                .resize_to(IMAGE_HEIGHT, IMAGE_WIDTH) \
                .conv(1, DE_).conv(3, DE_).conv(3, DE_) \
                .dropout(dropout_keep_rate, apply_dropout)   

    # Concatenate
    return full_layer.concat(layer_1_2, layer_1_4, layer_1_8) \
                     .conv(1, 64).conv(7, 64).conv(7, 64) \
                     .conv(3, 32).conv(3, 32) \
                     .incep_a() \
                     .dropout(dropout_keep_rate, apply_dropout) \
                     .conv(1, 32).conv(3, 32).conv(3, 16) \
                     .conv(1, NUM_LABELS, activation=None).tensor

def run_test(checkpoint_index='latest'):
    from engine import run_inference
    from config import TEST_DIR
    run_inference(model=make_model, image_dir=TEST_DIR, 
                  checkpoint_index=checkpoint_index)

if __name__ == "__main__":
    data = get_train_val_sets(image_dir=TRAIN_DIR)
    run_training(model=make_model, data=data)

