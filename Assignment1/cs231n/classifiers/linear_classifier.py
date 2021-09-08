import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier():

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate = 1e-3, reg = 1e-5, num_iters = 100, batch_size = 200, verbose = False):
        """
        使用 SGD 训练该线性分类器

        输入：
        - X：(N, D) 的训练数据
        - y：长度为 N 的标签数组
        - learning_rate：学习率
        - reg：正则化强度
        - num_iters：最优化时走的步数
        - batch_size：每一步训练的样本数
        - verbose：如果设置为 true，则在最优化时打印过程结果

        输出：
        一个列表，记录每次迭代的损失值
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
        
        # 使用 SGD 优化 W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # 按照 batch_size 对训练数据进行采样
            # 提示：使用 np.random.choice 产生序号；
            # 使用 replacement 的采样比没有使用 replacement 的采样速度更快
            batch_idx = np.random.choice(num_train, batch_size, replace = True)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            # 计算梯度和损失
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # 更新参数
            self.W += -learning_rate * grad

            if verbose and it %100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """ 使用训练好权重的线性分类器为测试数据预测标签 """
        y_pred = np.zeros(X.shape[1])

        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis = 1)

        return y_pred 
    
    def loss(self, X_batch, y_batch, reg):
        pass

class LinearSVM(LinearClassifier):
    """ 一个子类使用 Multiclass SVM loss """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class Softmax(LinearClassifier):
    """ 一个子类使用 Softmax + Cross-entropy 损失函数 """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vetorized(self.W, X_batch, y_batch, reg)