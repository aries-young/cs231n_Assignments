import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    结构化的 SVM 损失函数，简单的实现版本（带循环）

    输入是 D 维的，有 C 个类别，我们操作 minibatch，包含 N 个样本

    输入：
    - W：权重矩阵 D x C
    - X：minibatch N x D
    - y：标签数组 (N,)
    - reg：正则化强度

    返回：
    - 损失值 (float)
    - W 的梯度矩阵，大小和 W 一样
    """
    dw = np.zeros(W.shape)

    # 计算梯度和损失值
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]: continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dw[:,j] += X[i].T
                dw[:,y[i]] += -X[i].T
    
    # 现在我们得到了关于整个训练集的损失值的和
    # 但是我们想要的是一个平均值，所以拿他除以 num_train
    loss /= num_train
    dw /= num_train
    
    # 在数据损失上加上正则化
    loss += 0.5 * reg * np.sum(W * W)
    dw += reg * W

    return loss, dw

def svm_loss_vectorized(W, X, y, reg):
    """
    比起先计算损失再计算梯度，在计算损失的同时计算梯度可以更简单一些
    """
    loss = 0.0
    dw  = np.zeros(W.shape)

    # 向量化计算 SVM loss
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    correct_class_score = scores[range(num_train), list(y)].reshape(-1, 1)
    margins = np.maximum(0, scores - correct_class_score + 1)
    margins[range(num_train), list(y)] = 0
    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W *W)

    # 向量化计算梯度
    # 比起从头开始计算梯度，我们可以使用前面在计算损失时的一些东西
    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis = 1)

    dw = (X.T).dot(coeff_mat)
    dw = dw / num_train + reg * W

    return loss, dw