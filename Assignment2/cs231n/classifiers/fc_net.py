import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

class TwoLayerNet():
    """
    网络结构：affine - relu - affine - softmax

    注意：在 TwoLayerNet 中我们没有实现梯度下降
    我们会在 Solver 类中实现对应的 running optimization

    这个模型的学习参数存储在 self.param 参数字典中
    """

    def __init__(self, input_dim = 32 * 32 * 3, hidden_dim = 100, num_classes = 10, weight_scale = 1e-3, reg = 0.0):
        self.params = {}
        self.reg = reg

        # 初始化权重和偏置
        # 权重初始为一个高斯分布，标准差等于 weight_scale
        # 偏置初始化为 0
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y = None):
        """
        计算 minibatch 的损失和梯度
        """
        scores = None
        ar1_out, ar1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        a2_out, a2_cache = affine_forward(ar1_out, self.params['W2'], self.params['b2'])
        scores = a2_out

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        loss = loss + 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1']) + 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])
        dx2, dw2, db2 = affine_backward(dscores, a2_cache)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        grads['b2'] = db2
        dx1, dw1, db1 = affine_relu_backward(dx2, ar1_cache)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1'] = db1

        return loss, grads


class FullyConnectedNet():
    """
    N 层的全连接神经网络，网络结构为： {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    其中，batchnorm 和 dropout 作为可选项
    """
    
    def __init__(self, hidden_dims, input_dim = 32 * 32 * 3, num_classes = 10, dropout = 0, use_batchnorm = False, reg = 0.0, weight_scale = 1e-2, dtype = np.float32, seed = None):
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg 
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # 提醒，lightaime 源代码这里的初始化是错误的
        layer_input_dim = [input_dim] + hidden_dims
        for i, hd in enumerate(hidden_dims):
            self.params['W%d' % (i + 1)] = weight_scale * np.random.randn(layer_input_dim[i], hd)
            self.params['b%d' % (i + 1)] = weight_scale * np.zeros(hd)
            if self.use_batchnorm:
                self.params['gamma%d' % (i + 1)] = np.ones(hd)
                self.params['beta%d' % (i + 1)] = np.zeros(hd)
        self.params['W%d' % (self.num_layers)] = weight_scale * np.random.randn(layer_input_dim[-1], num_classes)
        self.params['b%d' % (self.num_layers)] = weight_scale * np.zeros(num_classes)

        # dropout 参数
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # batchnorm 参数
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y = None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # 将 batchnorm 和 dropout 的参数设置为 train/test
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        score = None
        layer_input = X
        ar_cache = {}
        dp_cache = {}

        for lay in range(self.num_layers - 1):
            if self.use_batchnorm:
                layer_input, ar_cache[lay] = affine_bn_relu_forward(layer_input, 
                                                                    self.params['W%d' % (lay + 1)], 
                                                                    self.params['b%d' % (lay + 1)],
                                                                    self.params['gamma%d' % (lay + 1)], 
                                                                    self.params['beta%d' % (lay + 1)], 
                                                                    self.bn_params[lay])
            else:
                layer_input, ar_cache[lay] = affine_relu_forward(layer_input, self.params['W%d' % (lay + 1)], self.params['b%d' % (lay + 1)])
            if self.use_dropout:
                layer_input, dp_cache[lay] = dropout_forward(layer_input, self.dropout_param)
        
        ar_out, ar_cache[self.num_layers] = affine_forward(layer_input, self.params['W%d' % (self.num_layers)], self.params['b%d' % (self.num_layers)])
        scores = ar_out

        if mode == 'test':
            return scores
        
        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)
        dhout = dscores
        loss = loss + 0.5 * self.reg * np.sum(self.params['W%d' % (self.num_layers)] * self.params['W%d' % (self.num_layers)])
        dx , dw , db = affine_backward(dhout , ar_cache[self.num_layers])
        grads['W%d' % (self.num_layers)] = dw + self.reg * self.params['W%d' % (self.num_layers)]
        grads['b%d' % (self.num_layers)] = db
        dhout = dx
        for idx in range(self.num_layers - 1):
            lay = self.num_layers - 1 - idx - 1
            loss = loss + 0.5 * self.reg * np.sum(self.params['W%d' % (lay + 1)] * self.params['W%d' % (lay + 1)])
            if self.use_dropout:
                dhout = dropout_backward(dhout, dp_cache[lay])
            if self.use_batchnorm:
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhout, ar_cache[lay])
            else:
                dx, dw, db = affine_relu_backward(dhout, ar_cache[lay])
            grads['W%d' % (lay + 1)] = dw + self.reg * self.params['W%d' % (lay + 1)]
            grads['b%d' % (lay + 1)] = db
            if self.use_batchnorm:
                grads['gamma%d' % (lay + 1)] = dgamma
                grads['beta%d' % (lay + 1)] = dbeta
            dhout = dx

        return loss, grads



