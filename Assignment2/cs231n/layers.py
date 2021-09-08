import numpy as np

def affine_forward(x, w, b):
    """
    计算全连接层的前向传播

    输入的 x 大小为 (N, d1, ..., dk)，包含一个由 N 样本组成的 minibatch
    每一个样本 x[i] 大小为 (d1, ..., dk)，我们将每一个输入压为一个维度 D = d1 * ... * dk 的向量
    然后将这个向量转变为一个维度为 M 的输出向量

    输入：
    - x：输入数据，大小为 (N, d1, ..., dk)
    - w：权重，大小为 (D, M)
    - b：偏差，大小为 (M,)

    返回：
    - out：输出结果，大小为 (N, M)
    - cache：(x, w, b)
    """
    out = None
    N = x.shape[0]
    x_rsp = x.reshape(N, -1)
    out = x_rsp.dot(w) + b

    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache) :
    """
    计算全连接层的反向传播

    输入：
    - dout：上游传回的导数信息，大小为(N, M)
    - cache：(x, w) 的元组

    返回元组：
    - dx：x 对应的梯度
    - dw：w 对应的梯度
    - db：b 对应的梯度
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    N = x.shape[0]
    x_rsp = x.reshape(N, -1)
    dx = dout.dot(w.T)
    dx = dx.reshape(*x.shape)
    dw = x_rsp.T.dot(dout)
    db = np.sum(dout, axis = 0)

    return dx, dw, db

def relu_forward(x) :
    """
    计算 ReLU 层的前向传播

    输入：
    - x：输入数据

    返回元组：
    - out：输出，大小和 x 一样
    - cache： x
    """
    out = None

    out = x * (x >= 0)

    cache = x
    return out, cache

def relu_backward(dout, cache) :
    """
    计算 ReLU 层的反向传播

    输入：
    - dout：上游传回的导数信息
    - cache：输入 x，大小和 dout 一样

    返回：
    - dx：x 对应的梯度
    """
    dx, x = None, cache

    dx = (x >= 0) * dout
    return dx

def batchnorm_forward(x, gamma, beta, bn_param) :
    """
    bacth normalization 的前向传播

    在训练的过程中，我们通过 minibatch 的统计数据得到样本的均值和方差，
    然后将它们用于标准化输入数据
    在训练的过程中，我们还保留每个特征均值和方差的指数衰减运行均值，
    这些均值会在测试时用于标准化数据

    在每一个时间步长，我们使用一个基于动量参数的指数衰减方式来更新期望和方差的运行时均值

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    batch normalization 的论文建议在测试时采用一个不同的方法：
    论文中作者是在大量训练图片的基础上为每个特征计算样本期望和方差的，而不是基于运行时的平均值
    而在我们的实现中，我们采用运行时的平均值来计算，原因是这样做不需要额外的估计步骤、
    在 torch7 中也是这样做的

    输入：
    - x：数据，大小为(N, D)
    - gamma：尺度参数，大小为(D, )
    - beta: 位移参数，大小为(D, )
    - bn_param：参数字典
    - mode：'train' or 'test'
    - eps：数值稳定的一个常数
    - momentum：动量常数
    - running_mean：大小为(D, ) 的数组，存储特征的运行时期望
    - running_var：大小为(D, ) 的数组，存储特征的运行时方差

    返回元组：
    - out：大小为(N, D)
    - cache: 一个元组存着反向传播时需要额度数据
    """
    mode = bn_param['mode']
    eps = bn_param.get('esp', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype = x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype = x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis = 0)
        sample_var = np.var(x, axis = 0)
        x_hat = (x - sample_mean) / (np.sqrt(sample_var + eps))
        out = gamma * x_hat + beta
        cache = (gamma, x, sample_mean, sample_var, eps, x_hat)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == 'test':
        scale = gamma / (np.sqrt(running_var + eps))
        out = x * scale + (beta - running_mean * scale)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout, cache) :
    """
    batch normalization 的反向传播

    输入：
    - dout：上游 传回的导数信息，大小为(N, D)
    - cache：batchnorm 向前传播的临时中间变量

    返回元组：
    - dx：对应 x 的梯度信息
    - dgamma：对应尺度参数 gamma 的梯度信息
    - dbeta：对应位移参数 beta 的梯度信息
    """
    dx, dgamma, dbeta = None, None, None

    gamma, x, u_b, sigma_squared_b, eps, x_hat = cache
    N = x.shape[0]

    dx_1 = gamma * dout
    dx_2_b = np.sum((x - u_b) * dx_1, axis = 0)
    dx_2_a = ((sigma_squared_b + eps) ** -0.5) * dx_1
    dx_3_b = (-0.5) * ((sigma_squared_b + eps) ** -1.5) * dx_2_b
    dx_4_b = dx_3_b * 1
    dx_5_b = np.ones_like(x) / N * dx_4_b
    dx_6_b = 2 * (x - u_b) * dx_5_b
    dx_7_a = dx_6_b * 1 + dx_2_a * 1
    dx_7_b = dx_6_b * 1 + dx_2_a * 1
    dx_8_b = -1 * np.sum(dx_7_b, axis = 0)
    dx_9_b = np.ones_like(x) / N * dx_8_b
    dx_10 = dx_9_b + dx_7_a

    dgamma = np.sum(x_hat * dout, axis = 0)
    dbeta = np.sum(dout, axis = 0)
    dx = dx_10

    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache) :
    """
    可选择的 batch normalization 反向传播
    """
    dx, dgamma, dbeta = None, None, None

    gamma, x, sample_mean, sample_var, eps, x_hat = cache
    N = x.shape[0]
    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * (x - sample_mean) * -0.5 * np.power(sample_var + eps, -1.5), axis = 0)
    dmean = np.sum(dx_hat * -1 / np.sqrt(sample_var + eps), axis = 0) + dvar * np.mean(-2 * (x - sample_mean), axis = 0)
    dx = 1 / np.sqrt(sample_var + eps) * dx_hat + dvar * 2.0 / N * (x - sample_mean) + 1.0 / N * dmean
    dgamma = np.sum(x_hat * dout, axis = 0)
    dbeta = np.sum(dout, axis = 0)

    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param) :
    """
    dropout 的前向传播

    输入：
    - x：输入数据
    - dropout_param：参数字典
    - p：我们按照 p 的概率对每个神经元失活
    - mode：'train' or 'test'，如果是 train，进行 dropout；如果是 test，直接返回输入
    - seed：随机数生成器的种子。传入这个参数是为了使函数的运行结果可复现，对梯度检查来说是必要的，但是在真实的网络中不需要

    输出：
    - out：和 x 一样的输出结果
    - cache：(dropout_param, mask) 的元组，在训练模式下，mask 是 dropout mask，用于乘以输入；在测试模式下， mask is None
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param :
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) >= p) / (1 - p)
        out = x * mask
    elif mode == 'test' :
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy = False)

    return out, cache

def dropout_backward(dout, cache) :
    """
    dropout 的反向传播

    输入：
    - dout：上游传回的导数信息
    - cache：(dropout_param, mask)
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']
    dx = None

    if mode == 'train':
        dx = dout * mask
    elif mode == 'test' :
        dx = dout
    return dx

def conv_forward_naive(x, w, b, conv_param) :
    """
    卷积层前向传播的一个简单实现

    输入包含 N 个数据点，每个数据点有 C 个通道，高 H、宽 W
    我们用 F 个不同的过滤器对每个输入进行卷积，每个过滤器都会滑过 C 个通道，高 HH、宽 WW

    输入：
    - x：输入数据，大小为(N, C, H, W)
    - w：过滤器的权重，大小为(F, C, HH, WW)
    - b：偏置，大小为(F, )
    - conv_param：参数字典
    - stride：步长
    - pad：填充

    返回：
    - out：输出数据，大小为(N, F, H', W')
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache：(x, w, b, conv_param)
    """
    out = None

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))

    x_pad = np.pad(x, ((0, ), (0, ), (pad, ), (pad, )), mode = 'constant', constant_values = 0)
    for i in range(H_out) :
        for j in range(W_out) :
            x_pad_masked = x_pad[:, :, i * stride : i * stride + HH, j * stride : j * stride + WW]
            for k in range(F) :
                out[:, k, i, j] = np.sum(x_pad_masked * w[k, :, : , : ], axis = (1, 2, 3))

    out = out + (b)[None, :, None, None]
    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache) :
    """
    卷积层反向传播的一个简单实现
    """
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    x_pad = np.pad(x, ((0, ), (0, ), (pad, ), (pad, )), mode = 'constant', constant_values = 0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db = np.sum(dout, axis = (0, 2, 3))
    x_pad = np.pad(x, ((0, ), (0, ), (pad, ), (pad, )), mode = 'constant', constant_values = 0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, : , i * stride : i * stride + HH, j * stride : j * stride + WW]
            for k in range(F):
                dw[k, :, : , : ] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis = 0)
            for n in range(N) :
                dx_pad[n, :, i * stride : i * stride + HH, j * stride : j * stride + WW] += np.sum((w[:, : , : , : ] * (dout[n, :, i, j])[:, None, None, None]), axis = 0)
    dx = dx_pad[:, : , pad : -pad, pad : -pad]
    
    return dx, dw, db
                
def max_pool_forward_naive(x, pool_param):
    """
    max poolig 层前向传播的简单实现

    输入：
    - x：输入数据，大小为 (N, C, H, W)
    - pool_param:
        - pool_height：pooling region 的高度
        - pool_width：pooling region 的宽度
        - stride：步长

    返回元组：
    - out：输出数据
    - cache：(x, pool_param)
    """
    out = None
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = (H - HH) // stride + 1
    W_out = (W - WW) // stride + 1
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:, :, i * stride : i * stride + HH, j * stride : j * stride + WW]
            out[:, :, i, j] = np.max(x_masked, axis = (2, 3))

    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    max pooling 层反向传播的简单实现
    """
    dx = None
    x, pool_param = cache
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = (H - HH) // stride + 1
    W_out = (W - WW) // stride + 1
    dx = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:, :, i * stride : i * stride + HH, j * stride : j * stride + WW]
            max_x_masked = np.max(x_masked, axis = (2, 3))
            temp_binary_mask = (x_masked == (max_x_masked)[:, :, None, None])
            dx[:, :, i * stride : i * stride + HH, j * stride : j *stride + WW] += temp_binary_mask * (dout[:, :, i, j])[:, :, None, None]

    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    spatial batch normalization 的前向传播

    输入：
    - x：输入数据，大小为 (N, C, H, W)
    - gamma：尺度参数，大小为 (C,)
    - beta：位移参数，大小为 (C,)
    - bn_param：
        - mode：train or test
        - eps：数值稳定的一个常数
        - momentum：计算 running mean / variance 时用到的一个常数
                    momentum = 0 表示在每一步中原来的信息会被完全丢弃
                    momentum = 1 表示新信心永远不会被纳入计算，默认 momentum = 0.9，在大多数情况下是表项良好的
        - running_mean：大小为 (D,)
        - running_var：大小为 (D,)
    
    返回元组：
    - out：输出结果，大小为 (N, C, H, W)
    - cache：反向传播需要的值
    """
    out, cache = None, None
    N, C, H, W = x.shape
    temp_output, cache = batchnorm_forward(x.transpose(0, 3, 2, 1).reshape((N * H * W , C)), gamma, beta, bn_param)
    out = temp_output.reshape(N, W, H, C).transpose(0, 3, 2, 1)

    return out, cache

def spatial_batchnorm_backward(dout, cache):
    """
    spatial batch normalization 的反向传播
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape
    dx_temp, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0, 3, 2, 1).reshape((N * H * W, C)), cache)
    dx = dx_temp.reshape(N, W, H, C).transpose(0, 3, 2, 1)
    return dx, dgamma, dbeta

def svm_loss(x, y):
    """
    计算 SVM 的损失和梯度

    输入：
    - x：输入数据，大小为 (N, C)，其中 x[i, j] 表示第 i 个数据点对应第 j 类的得分
    - y：标签数组

    返回元组：
    - loss：损失值
    - dx：梯度值
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis = 1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N

    return loss, dx

def softmax_loss(x, y):
    """
    计算 softmax 的损失和梯度
    """
    probs = np.exp(x - np.max(x, axis = 1, keepdims = True))
    probs /= np.sum(probs, axis = 1, keepdims = True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx
    
