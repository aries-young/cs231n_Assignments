import numpy as np
import h5py

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class PretrainedCNN(object):
  def __init__(self, dtype=np.float32, num_classes=100, input_size=64, h5_file=None):
    self.dtype = dtype
    self.conv_params = []
    self.input_size = input_size
    self.num_classes = num_classes
    
    # TODO: In the future it would be nice if the architecture could be loaded from
    # the HDF5 file rather than being hardcoded. For now this will have to do.
    self.conv_params.append({'stride': 2, 'pad': 2})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 2, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 2, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 2, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 2, 'pad': 1})

    self.filter_sizes = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    self.num_filters = [64, 64, 128, 128, 256, 256, 512, 512, 1024]
    hidden_dim = 512

    self.bn_params = []
    
    cur_size = input_size
    prev_dim = 3
    self.params = {}
    for i, (f, next_dim) in enumerate(zip(self.filter_sizes, self.num_filters)):
      fan_in = f * f * prev_dim
      self.params['W%d' % (i + 1)] = np.sqrt(2.0 / fan_in) * np.random.randn(next_dim, prev_dim, f, f)
      self.params['b%d' % (i + 1)] = np.zeros(next_dim)
      self.params['gamma%d' % (i + 1)] = np.ones(next_dim)
      self.params['beta%d' % (i + 1)] = np.zeros(next_dim)
      self.bn_params.append({'mode': 'train'})
      prev_dim = next_dim
      if self.conv_params[i]['stride'] == 2: cur_size /= 2
    
    # Add a fully-connected layers
    fan_in = int(cur_size * cur_size * self.num_filters[-1])
    self.params['W%d' % (i + 2)] = np.sqrt(2.0 / fan_in) * np.random.randn(fan_in, hidden_dim)
    self.params['b%d' % (i + 2)] = np.zeros(hidden_dim)
    self.params['gamma%d' % (i + 2)] = np.ones(hidden_dim)
    self.params['beta%d' % (i + 2)] = np.zeros(hidden_dim)
    self.bn_params.append({'mode': 'train'})
    self.params['W%d' % (i + 3)] = np.sqrt(2.0 / hidden_dim) * np.random.randn(hidden_dim, num_classes)
    self.params['b%d' % (i + 3)] = np.zeros(num_classes)
    
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

    if h5_file is not None:
      self.load_weights(h5_file)

  
  def load_weights(self, h5_file, verbose=False):
    """
    从 HDF5 文件加载预训练好的权重

    输入：
    - h5_file：HDF5 文件路径
    - verbose：是否打印 debug 信息
    """

    # Before loading weights we need to make a dummy forward pass to initialize
    # the running averages in the bn_pararams
    x = np.random.randn(1, 3, self.input_size, self.input_size)
    y = np.random.randint(self.num_classes, size=1)
    loss, grads = self.loss(x, y)

    with h5py.File(h5_file, 'r') as f:
      for k, v in f.items():
        v = np.asarray(v)
        if k in self.params:
          if verbose: print(k, v.shape, self.params[k].shape)
          if v.shape == self.params[k].shape:
            self.params[k] = v.copy()
          elif v.T.shape == self.params[k].shape:
            self.params[k] = v.T.copy()
          else:
            raise ValueError('shapes for %s do not match' % k)
        if k.startswith('running_mean'):
          i = int(k[12:]) - 1
          assert self.bn_params[i]['running_mean'].shape == v.shape
          self.bn_params[i]['running_mean'] = v.copy()
          if verbose: print(k, v.shape)
        if k.startswith('running_var'):
          i = int(k[11:]) - 1
          assert v.shape == self.bn_params[i]['running_var'].shape
          self.bn_params[i]['running_var'] = v.copy()
          if verbose: print(k, v.shape)
        
    for k, v in self.params.items():
      self.params[k] = v.astype(self.dtype)

  
  def forward(self, X, start=None, end=None, mode='test'):
    """
    计算模型前向传播的一部分，可以在任意层开始和结束，既可以在训练模型，也可以在测试模型
    我们可以在开始层传入任意的输入数据，然后在结束层得到对应的输出
    cache 对象被用来存储开始层到结束层反向传播需要的变量信息

    在这里，一层被定义为如下的结构：
    [conv - spatial batchnorm - relu] (deep CNN model 中有 9 层这样的结构)
    [affine - batchnorm - relu] (deep CNN model 中有 1 层这样的结构)
    [affine] (deep CNN model 中有 9 层这样的结果)

    输入：
    - X：开始层的输入数据，如果开始层是第 0 层，那么输入数据的大小为 (N, C, 64, 64)
    - start：开始层的序号，start = 0 从第一个卷积层开始
    - end：结束层的序号，start = 11 代表最后一个全连接层
    - mode：'train' or 'test'，目的是 bn 在训练时和测试时有着不同的行为
    
    输出：
    - out：结束层的输出结果
    - cache：同上面描述
    """
    X = X.astype(self.dtype)
    if start is None: start = 0
    if end is None: end = len(self.conv_params) + 1
    layer_caches = []

    prev_a = X
    for i in range(start, end + 1):
      i1 = i + 1
      if 0 <= i < len(self.conv_params):
        # This is a conv layer
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        gamma, beta = self.params['gamma%d' % i1], self.params['beta%d' % i1]
        conv_param = self.conv_params[i]
        bn_param = self.bn_params[i]
        bn_param['mode'] = mode

        next_a, cache = conv_bn_relu_forward(prev_a, w, b, gamma, beta, conv_param, bn_param)
      elif i == len(self.conv_params):
        # This is the fully-connected hidden layer
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        gamma, beta = self.params['gamma%d' % i1], self.params['beta%d' % i1]
        bn_param = self.bn_params[i]
        bn_param['mode'] = mode
        next_a, cache = affine_bn_relu_forward(prev_a, w, b, gamma, beta, bn_param)
      elif i == len(self.conv_params) + 1:
        # This is the last fully-connected layer that produces scores
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        next_a, cache = affine_forward(prev_a, w, b)
      else:
        raise ValueError('Invalid layer index %d' % i)

      layer_caches.append(cache)
      prev_a = next_a

    out = prev_a
    cache = (start, end, layer_caches)
    return out, cache


  def backward(self, dout, cache):
    """
    计算 forward 选中层的反向传播

    输入：
    - dout：结束层的梯度，大小同 forward 函数返回的 out 变量
    - cache：forward 函数返回的 cache 对象

    返回：
    - dX：开始层的梯度，大小同 forward 函数输入的 X 变量
    - grads：所有参数的梯度，grads 字典作为 self.params 的子集，grads[k] 和 self.params[k] 有着相同的大小
    """
    start, end, layer_caches = cache
    dnext_a = dout
    grads = {}
    for i in reversed(range(start, end + 1)):
      i1 = i + 1
      if i == len(self.conv_params) + 1:
        # This is the last fully-connected layer
        dprev_a, dw, db = affine_backward(dnext_a, layer_caches.pop())
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
      elif i == len(self.conv_params):
        # This is the fully-connected hidden layer
        temp = affine_bn_relu_backward(dnext_a, layer_caches.pop())
        dprev_a, dw, db, dgamma, dbeta = temp
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
        grads['gamma%d' % i1] = dgamma
        grads['beta%d' % i1] = dbeta
      elif 0 <= i < len(self.conv_params):
        # This is a conv layer
        temp = conv_bn_relu_backward(dnext_a, layer_caches.pop())
        dprev_a, dw, db, dgamma, dbeta = temp
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
        grads['gamma%d' % i1] = dgamma
        grads['beta%d' % i1] = dbeta
      else:
        raise ValueError('Invalid layer index %d' % i)
      dnext_a = dprev_a

    dX = dnext_a
    return dX, grads


  def loss(self, X, y=None):
    """
    用于训练网络的分类器损失
    """
    # Note that we implement this by just caling self.forward and self.backward
    mode = 'test' if y is None else 'train'
    scores, cache = self.forward(X, mode=mode)
    if mode == 'test':
      return scores
    loss, dscores = softmax_loss(scores, y)
    dX, grads = self.backward(dscores, cache)
    return loss, grads