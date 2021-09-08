import numpy as np
from random import shuffle 

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax 损失函数的简单实现

    输入 D 维，有 C 个类，操作 minibatch 有 N 个样本

    输入：
    - W：(D, C) 的权重矩阵
    - X：(N, D) 的 minibatch
    - y：长度为 N 的标签数组
    - reg: 正则化强度

    返回：
    - 损失值 (float)
    - 梯度矩阵
    """

    # 初始化损失和梯度
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        shift_scores = scores - max(scores)
        loss_i = -shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
        loss += loss_i
        for j in range(num_classes):
            softmax_output = np.exp(shift_scores[j]) / sum(np.exp(shift_scores))
            if j == y[i]:
                dW[:,j] += (-1 + softmax_output) * X[i]
            else:
                dW[:,j] += softmax_output * X[i]
        
    
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W

    return loss, dW

def softmax_loss_vetorized(W, X, y, reg):
    
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    shift_scores = scores - np.max(scores, axis = 1).reshape(-1, 1)
    softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis = 1).reshape(-1, 1)
    loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dS = softmax_output.copy()
    dS[range(num_train), list(y)] += -1
    dW = (X.T).dot(dS)
    dW = dW / num_train + reg * W

    return loss, dW