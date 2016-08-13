from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from functools import reduce
from math import ceil
import tensorflow as tf
from .variable_registry import registered_variable

from six.moves import range, zip

@registered_variable()
def tf_weight(*args, **kwargs):
    return tf.Variable(*args, **kwargs)

@registered_variable(is_weight=False)
def tf_bias(*args, **kwargs):
    return tf.Variable(*args, **kwargs)

def make_patch(size, in_depth, out_depth, std=0.1):
    patch = size if isinstance(size, tuple) else (size, size)
    return tf_weight(tf.truncated_normal(
                     [patch[0], patch[1], in_depth, out_depth], 
                     stddev=std))

def make_bias(depth, zeros=True):
    return tf_bias(tf.zeros([depth])) if zeros \
           else tf_bias(tf.ones([depth]))

def make_conv(in_data, patch_size, depth, 
              stride=1, pad='SAME', std=0.1, activation='tanh'):
    in_depth = in_data.get_shape().as_list()[3]
    conv = tf.nn.conv2d(in_data, 
                        make_patch(patch_size, in_depth, depth, std=std), 
                        [1, stride, stride, 1], 
                        padding=pad)
    acts = {
        'relu': lambda: tf.nn.relu(conv + make_bias(depth)),
        'tanh': lambda: tf.nn.tanh(conv + make_bias(depth)),
        'sigmoid': lambda: tf.sigmoid(conv + make_bias(depth))
    }
    return conv if activation is None else acts[activation]()

def _make_pool(pool_func, in_data, patch_size, stride=1, pad='SAME'):
    patch = patch_size if isinstance(patch_size, tuple) else \
            (patch_size, patch_size)
    return pool_func(in_data, 
                      [1, patch[0], patch[1], 1], 
                      [1, stride, stride, 1], 
                      padding=pad)

def make_max(in_data, patch_size, stride=1, pad='SAME'):
    return _make_pool(tf.nn.max_pool, in_data, patch_size,
                      stride=stride, pad=pad)

def make_avg(in_data, patch_size, stride=1, pad='SAME'):
    return _make_pool(tf.nn.avg_pool, in_data, patch_size,
                      stride=stride, pad=pad)

# Make fully connected weights and biases
def make_fc_w_b(feature_size, out_c, hidden_layers=[], std=0.1):
    cc = [feature_size] + hidden_layers + [out_c]
    bb = list(zip(cc[:-1], cc[1:]))
    return([tf_weight(tf.truncated_normal([r, c], stddev=std)) for r, c in bb],
           [tf_bias(tf.zeros([c])) for _, c in bb])

# Make fully connected
def make_fc(in_data, out_c, hidden=[], drop_out=False, std=0.1):
    # in_data expects a 2 dimensional tensor with shape [N, D]
    features = in_data.get_shape().as_list()[1]
    weights, biases = make_fc_w_b(features, out_c, hidden, std=std)
    
    lc = lambda x, wb: tf.matmul(x, wb[0]) + wb[1]
    r = lambda x, wb: tf.nn.relu(lc(x, wb))
    d_o = lambda x, wb: tf.nn.dropout(r(x, wb), 0.5)

    w_b =  list(zip(weights, biases))
    assert len(w_b) > 0, "Expected at least 1 set of weights and biases."

    rd = d_o if drop_out else r
    h = in_data if len(w_b) == 1 else \
        reduce(lambda t, wb: rd(t, wb), w_b[1:-1], rd(in_data, w_b[0]))

    return lc(h, w_b[-1])

def make_lstm_w_b(in_features, num_nodes, std=0.1):
    x = tf_weight(tf.truncated_normal([in_features, num_nodes], stddev=std))
    m = tf_weight(tf.truncated_normal([num_nodes, num_nodes], stddev=std))
    b = tf_bias(tf.zeros([1, num_nodes]))
    return x, m, b

class TensorOp():
    def __init__(self, tensor):
        self.tensor = tensor
        
    def __add__(self, other):
        return TensorOp(self.tensor + other.tensor)
    
    def __radd__(self, other):
        return TensorOp(self.tensor + other.tensor)
    
    @property
    def shape(self):
        return self.tensor.get_shape().as_list()
    
    def sigmoid(self, add_bias=False):
        return TensorOp(tf.nn.sigmoid(self.tensor)) if not add_bias else \
               TensorOp(tf.nn.sigmoid(self.tensor + make_bias(self.shape[-1])))
    
    def tanh(self, add_bias=False):
        return TensorOp(tf.nn.tanh(sef.tensor)) if not add_bias else \
               TensorOp(tf.nn.tanh(self.tensor + make_bias(self.shape[-1])))
    
    def add_bias(self, depth=None, zeros=True):
        d = depth if depth is not None else self.shape[-1]
        return TensorOp(self.tensor + make_bias(d, zeros=zeros))
    
    def concat(self, *args, **kwargs):
        # kwargs may include 'dim'
        dim = kwargs.get('dim', None)
        concat_dim = dim if dim is not None else len(self.shape) - 1
        return TensorOp(tf.concat(concat_dim, [self.tensor] + \
               [t.tensor for t in args]))
        
    def conv(self, patch_size, depth, stride=1, 
             pad='SAME', std=0.1, activation='tanh'):
        t = make_conv(self.tensor, patch_size, depth, stride=stride, 
                      pad=pad, std=std, activation=activation)
        return TensorOp(t)
    
    def max_pool(self, patch_size, stride=1, pad='SAME'):
        t = make_max(self.tensor, patch_size, stride=stride, pad=pad)
        return TensorOp(t)
    
    def avg_pool(self, patch_size, stride=1, pad='SAME'):
        t = make_avg(self.tensor, patch_size, stride=stride, pad=pad)
        return TensorOp(t)
    
    def dropout(self, keep_prob, apply=True):
        return TensorOp(tf.nn.dropout(self.tensor, keep_prob)) \
               if apply else self
    
    def to_2d(self):
        shape = self.tensor.get_shape().as_list()
        t = tf.reshape(self.tensor, 
                       [shape[0], reduce(lambda a, b: a * b, shape[1:])])
        return TensorOp(t)

    def fully_conn(self, out_units, hidden=[], std=0.1 ):
        t = make_fc(self.tensor, out_units, 
                    hidden=hidden, drop_out=False, std=std)
        return TensorOp(t)
    
    def resize_to(self, height=None, width=None, scale=None):
        h, w = (height, width) if scale is None else \
               [ceil(v * scale) for v in self.shape[1:3]]
        return TensorOp(tf.image.resize_images(self.tensor, int(h), int(w)))
    
    def grid_reduce_a(self, d33=384, dt=(64, 96, 96)):
        conv33 = self.conv(3, d33, stride=2, pad='VALID')
        tower = self.conv(1, dt[0]).conv(3, dt[1]) \
                    .conv(3, dt[2], stride=2, pad='VALID')
        max_p = self.max_pool(3, stride=2, pad='VALID') 
        t = tf.concat(3, [op.tensor for op in [conv33, tower, max_p]], 
                      name="GridReduce_a")
        return TensorOp(t)
    
    def incep_a(self, d11=64, d55=(48, 64), d33=(64, 96,96), dta=64):
        conv11 = self.conv(1, d11)
        conv55 = self.conv(1, d55[0]).conv(5, d55[1])
        conv33 = self.conv(1, d33[0]).conv(3, d33[1]).conv(3, d33[2])
        avg_p = self.avg_pool(3).conv(1, dta)
        t = tf.concat(3, [op.tensor for op in
                             [conv11, conv55, conv33, avg_p]],
                      name="Inception_a")
        return TensorOp(t)
    
    def grid_reduce_b(self, dt1=(192, 320), dt2=(192, 192, 192, 192)):
        tower1 = self.conv(1, dt1[0]) \
                     .conv(3, dt1[1], stride=2, pad='VALID')
        tower2 = self.conv(1, dt2[0]) \
                     .conv((1,7), dt2[1]) \
                     .conv((7,1), dt2[2]) \
                     .conv(3, dt2[3], stride=2, pad='VALID')
        max_p = self.max_pool(3, stride=2, pad='VALID')
        t = tf.concat(3, [op.tensor for op in [tower1, tower2, max_p]], 
                      name="GridReduce_b")
        return TensorOp(t)
    
    def incep_b(self, d11=192, dt1=(160, 160, 192), 
                dt2=(160, 160, 160, 160, 192),
                dta=192):
        conv11 = self.conv(1, d11)
        tower1 = self.conv(1, dt1[0]) \
                     .conv((1,7), dt1[1]) \
                     .conv((7,1), dt1[2])
        tower2 = self.conv(1, dt2[0]) \
                     .conv((7,1), dt2[1]) \
                     .conv((1,7), dt2[2]) \
                     .conv((7,1), dt2[3]) \
                     .conv((1,7), dt2[4])
        avg_p = self.avg_pool(3).conv(1, dta)
        t = tf.concat(3, [op.tensor for op in
                             [conv11, tower1, tower2, avg_p]],
                      name="Inception_b")
        return TensorOp(t)
    
    def incep_c(self, d11=320, 
                dt1=(384, 384, 384),
                dt2=(448, 384, 384, 384),
                dta=192):
        conv11 = self.conv(1, d11)
        
        tower1 = self.conv(1, dt1[0])
        tower11 = tower1.conv((1,3), dt1[1])
        tower12 = tower1.conv((3,1), dt1[2])
        
        tower2 = self.conv(1, dt2[0]).conv(3, dt2[1])
        tower21 = tower2.conv((1,3), dt2[2])
        tower22 = tower2.conv((3,1), dt2[3])
        
        avg_p = self.avg_pool(3).conv(1, dta)
        
        t = tf.concat(3, [op.tensor for op in
                             [conv11,
                              tower11, tower12,
                              tower21, tower22,
                              avg_p]],
                      name="Inception_c")
        return TensorOp(t)
