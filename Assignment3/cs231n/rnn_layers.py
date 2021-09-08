import numpy as np

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    普通 RNN 的单个时间步前向传播，使用 tanh 激活函数

    输入数据的维度为 D，hidden state 的维度为 H，minibatch 的大小为 N

    输入：
    - x：当前时间步的输入，大小为 (N, D)
    - prev_h：上一时间步的 hidden state，大小为 (N, H)
    - Wx：input-to-hidden 连接的权重矩阵，大小为 (D, H)
    - Wh：hidden-to-hidden 连接的权重矩阵，大小为 (H, H)
    - b：偏置，大小为 (H,)

    返回元组：
    - next_h：下一个 hidden state，大小为 (N, H)
    - cache：存储反向传播所需要的值
    """
    next_h, cache = None, None

    a = prev_h.dot(Wh) + x.dot(Wx) + b
    next_h = np.tanh(a)
    cache = (x, prev_h, Wh, Wx, b, next_h)

    return next_h, cache

def rnn_step_backward(dnext_h, cache):
    """普通 RNN 的单个时间步反向传播"""
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None

    # 可以根据 tanh 输出的值计算局部导数
    x, prev_h, Wh, Wx, b, next_h = cache
    da = dnext_h * (1 - next_h * next_h)
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    db = np.sum(da, axis = 0)

    return dx, dprev_h, dWx, dWh, db

def rnn_forward(x, h0, Wx, Wh, b):
    """
    对整个数据序列执行普通 RNN 的前向传播
    我们假设输入序列由 T 个 D 维的向量组成
    RNN 的 hidden size = H，运行在大小为 N 的minibatch 上
    运行完前向传播后，我们返回所有时间步的 hidden states

    输入：
    - h0：初始的 hidden state，大小为 (H,)

    返回：
    - h：整个时间序列的 hidden states，大小为 (N, T, H)
    """
    h, cache = None, None

    N, T, D = x.shape
    (H, ) = b.shape
    h = np.zeros((N, T, H))
    prev_h = h0
    for t in range(T):
        xt = x[:, t, :]
        next_h, _ = rnn_step_forward(xt, prev_h, Wx, Wh, b)
        prev_h = next_h
        h[:, t, :] = prev_h
    cache = (x, h0, Wh, Wx, b, h)

    return h, cache

def rnn_backward(dh, cache):
    """
    计算 RNN 在整个数据序列上的反向传播

    输入：
    - h：上游传回的所有 hidden states 的梯度信息，大小为 (N, T, H)
    """
    x, h0, Wh, Wx, b, h = cache
    N, T, H = dh.shape
    _, _, D = x.shape

    next_h = h[:, T - 1, :]
    dprev_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx= np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))

    for t in range(T):
        t = T-1-t
        xt = x[:,t,:]
    
        if t ==0:
            prev_h = h0
        else:
            prev_h = h[:,t-1,:]
        
        step_cache = (xt, prev_h, Wh, Wx, b, next_h)
        next_h = prev_h
        dnext_h = dh[:,t,:] + dprev_h
        dx[:,t,:], dprev_h, dWxt, dWht, dbt = rnn_step_backward(dnext_h, step_cache)
        dWx, dWh, db = dWx+dWxt, dWh+dWht, db+dbt
    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
    """
    word embeddings 的前向传播
    我们在大小为 N 的 minibatch 进行操作，其中每个序列的长度为 T
    我们假设单词表有 V 个单词，每个单词被赋值给一个维度为 D 的向量

    输入：
    - x：大小为 (N, T) 的单词索引数组，x 中的每个元素 idx 在范围 0 <= idx < V 的范围内
    - W：大小为 (V, D) 的权重矩阵，给出了所有单词的单词向量

    返回：
    - out：大小为 (N, T, D) 的数组，给出所有输入单词额单词向量
    - cache：存储用于反向传播的值
    """
    out, cache = None, None
    N, T = x.shape
    V, D = W.shape
    out = np.zeros((N, T, D))
  
    for i in range(N):
        for j in range(T):
            out[i, j] = W[x[i,j]]
  
    cache = (x, W.shape)

    return out, cache

def word_embedding_backward(dout, cache):
    """
    word embeddings 的反向传播
    我们不能反向传播到单词中，因为它们是整数
    所以，我们只返回 word embedding 矩阵的梯度
    """
    dW = None

    x, W_shape = cache
    dW = np.zeros(W_shape)
    np.add.at(dW, x, dout)

    return dW

def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    
    return top / (1 + z)

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    LSTM 的单个时间步前向传播

    输入：
    - x：大小为 (N, D) 的输入
    - prev_h：前面的 hidden state，大小为 (N, H)
    - prev_c：前面的 cell state，大小为 (N, H)
    - Wx：input-to-hidden 权重，大小为 (D, 4H)
    - Wh：hidden-to-hidden 权重，大小为 (H, 4H)
    - b：偏置，大小为 (4H,)

    返回：
    - next_h：下一 hidden state，大小为 (N, H)
    - next_c：下一 cell state，大小为 (N, H)
    - cache：存储反向传播需要的值
    """
    next_h, next_c, cache = None, None, None

    H = Wh.shape[0]

    a = x.dot(Wx) + prev_h.dot(Wh) + b

    z_i = sigmoid(a[:,:H])
    z_f = sigmoid(a[:,H:2*H])
    z_o = sigmoid(a[:,2*H:3*H])
    z_g = np.tanh(a[:,3*H:])

    next_c = z_f * prev_c + z_i * z_g
    z_t = np.tanh(next_c)
    next_h = z_o * z_t

    cache = (z_i, z_f, z_o, z_g, z_t, prev_c, prev_h, Wx, Wh, x)

    return next_h, next_c, cache 

def lstm_step_backward(dnext_h, dnext_c, cache):
    """ LSTM 的单个时间步反向传播 """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None

    H = dnext_h.shape[1]
    z_i, z_f, z_o, z_g, z_t, prev_c, prev_h, Wx, Wh, x = cache
  
    dz_o = z_t * dnext_h
    dc_t = z_o * (1 - z_t * z_t) * dnext_h + dnext_c
    dz_f = prev_c * dc_t
    dz_i = z_g * dc_t
    dprev_c = z_f * dc_t
    dz_g = z_i * dc_t
    
    da_i = (1 - z_i) * z_i * dz_i
    da_f = (1 - z_f) * z_f * dz_f
    da_o = (1 - z_o) * z_o * dz_o
    da_g = (1 - z_g * z_g) * dz_g
    da = np.hstack((da_i, da_f, da_o, da_g))
  
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    
    db = np.sum(da, axis = 0)
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)

    return dx, dprev_h, dprev_c, dWx, dWh, db

def lstm_forward(x, h0, Wx, Wh, b):
    """
    LSTM 对整个数据序列的前向传播
    我们假设每一个输入序列由 T 个 D 维的向量组成
    LSTM 的 hidden size = H，我们在 minibatch 上计算前向传播
    每个 minibatch 含 N 个序列
    前向传播结束，我们返回所有时间步的 hidden states

    注意：初始的 cell state 被当做输入传入，但是初始时它被置为 0
    其次， cell state 是不被返回的，它作为 LSTM 的内部变量，从外部无法获得
    """
    h, cache = None, None

    N, T, D = x.shape
    H = b.shape[0] // 4
    h = np.zeros((N, T, H))
    cache = {}
    prev_h = h0
    prev_c = np.zeros((N, H))

    for t in range(T):
        xt = x[:, t, :]
        next_h, next_c, cache[t] = lstm_step_forward(xt, prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c
        h[:, t, :] = prev_h

    return h, cache

def lstm_backward(dh, cache):
    """ LSTM 对真个数据序列的反向传播 """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    N, T, H = dh.shape
    _i, z_f, z_o, z_g, z_t, prev_c, prev_h, Wx, Wh, x = cache[T-1]
    D = x.shape[1]
    
    dprev_h = np.zeros((N, H))
    dprev_c = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx= np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))
    
    for t in range(T):
        t = T-1-t
        step_cache = cache[t]
        dnext_h = dh[:,t,:] + dprev_h
        dnext_c = dprev_c
        dx[:,t,:], dprev_h, dprev_c, dWxt, dWht, dbt = lstm_step_backward(dnext_h, dnext_c, step_cache)
        dWx, dWh, db = dWx+dWxt, dWh+dWht, db+dbt
    
    dh0 = dprev_h  
    
    return dx, dh0, dWx, dWh, db

def temporal_affine_forward(x, w, b):
    """
    temporal affine layer 的前向传播。输入是一组维度为 D 的向量，
    排列成 N 个时间序列的 minibatch，每一个 minibatch 的长度为 T
    我们使用全连接网络将每一个向量转化成一个维度为 M 的输出向量

    输入：
    - x：大小为 (N, T, D) 的输入数据
    - w：大小为 (D, M) 的权重
    - b：大小为 (M,) 的偏置

    返回：
    - out：大小为 (N, T, M) 的输出数据
    - cache
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
  
    return out, cache

def temporal_affine_backward(dout, cache):
    """ temppral affine layer 的反向传播 """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db

def temporal_softmax_loss(x, y, mask, verbose = False):
    """
    时序版本的 softmax loss 用在 RNN 中。假设我们对大小为 V 的单词表
    在长度为 T 的时间序列中的每个时间步都进行预测，对大小为 N 的 
    minibatch 进行预测。输入 x 给出了所有时间步上每个单词元素的得分，
    y 给出了每个时间步上 ground-truth element 的索引。在每个时间步，我们
    使用交叉熵损失，计算所有时间步损失的总和， 并对 minibatch 求平均

    另外一个复杂的问题是，不同长度的序列可能会被用来组合成一个 minibatch，并用 NULL 进行填充，
    因此，我们需要在某些时间步上忽略模型的输出。可选参数 mask 可以让我们决定那些元素应该加入
    到损失当中，哪些应该忽略

    输入：
    - x：大小为 (N, T, V) 的得分输入
    - y：ground-truth indices，大小为 (N, T)
    - mask：大小为 (N, T) 的布尔数组，mask[i, t] 决定 x[i, t] 是否应该被纳入损失

    返回：
    - loss
    - dx
    """
    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)
  
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]
  
    if verbose: print('dx_flat: ', dx_flat.shape)
  
    dx = dx_flat.reshape(N, T, V)
  
    return loss, dx