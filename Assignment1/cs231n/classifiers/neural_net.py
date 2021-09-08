import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet():
    """
    两层的全连接神经网络

    网络的输入为 N 维，隐藏层为 H 维，在 C 个类别上进行分类
    我们用 softmax + L2 训练网络，在第一个全连接层后使用 ReLU 激活

    网络结构如下：
    input - FC layer - ReLU - FC layer - softmax 
    """
    def __init__(self, input_size, hidden_size, output_size, std = 1e-4):
        """
        初始化模型

        权重初始化为一些很小的随机值，偏置初始化为 0
        权重和偏置都存储在 self.params 中，self.param 是一个字典包含如下的键值：
        - W1：第一层的权重 (D, H)
        - b1：第一层的偏置 (H,)
        - W2：第二层的权重 (H, C)
        - b2：第二层的偏置 (C,)
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y = None, reg = 0.0):
        """
        计算两层全连接层网络的损失和梯度

        输入：
        - X：输入数据 (N, D)
        - y：训练数据的标签
             这是一个可选的参数，如果没有这个参数传入，我们就只返回得分；如果传入，就返回损失和梯度
        - reg：正则化强度

        输出：
        如果 y 为空，则返回一个 (N, C) 的得分矩阵，scores[i, c] 是 X[i] 在 c 类上的得分
        如果 y 不空，则返回当前 batch 的损失和梯度
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # 计算前向传播
        scores = None
        h_output = np.maximum(0, X.dot(W1) + b1)
        scores = h_output.dot(W2) + b2

        if y is None:
            return scores
        
        loss = None 
        # 结束前向传播，计算损失
        # 包括 W1 和 W2 的数据损失 + L2 正则化，使用 softmax 损失函数
        # 为了使我们的结果和课程的结果一致，要求在正则化损失前面乘上系数 0.5
        shift_scores = scores - np.max(scores, axis = 1).reshape(-1, 1)
        softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis = 1).reshape(-1, 1)
        loss = -np.sum(np.log(softmax_output[range(N), list(y)]))
        loss /= N
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # 计算反向传播
        grads = {}
        dscores = softmax_output.copy()
        dscores[range(N), list(y)] -= 1
        dscores /= N
        grads['W2'] = h_output.T.dot(dscores) + reg * W2
        grads['b2'] = np.sum(dscores, axis = 0)

        dh = dscores.dot(W2.T)
        dh_ReLu = (h_output > 0) * dh
        grads['W1'] = X.T.dot(dh_ReLu) + reg * W1
        grads['b1'] = np.sum(dh_ReLu, axis = 0)

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate = 1e-3,
              learning_rate_decay = 0.95,
              reg = 1e-5,
              num_iters = 100,
              batch_size = 200,
              verbose = False):
        """
        使用 SGD 训练网络
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # 使用 SGD 优化参数
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # 产生 minibatch
            idx = np.random.choice(num_train, batch_size, replace = True)
            X_batch = X[idx]
            y_batch = y[idx]

            # 使用当前的 minibatch 计算损失和梯度
            loss, grads = self.loss(X_batch, y = y_batch, reg = reg)
            loss_history.append(loss)

            # 使用梯度字典中的梯度信心更新参数
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            
            # 每一次迭代，检查训练集的准确率、验证集的准确率和衰退的学习率
            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }
    
    def predict(self, X):
        """
        使用训练好权重的两层神经网络为测试集打上标签
        我们为每个测试数据预测 C 个类的评分，将测试数据分配给得分最高的类
        """
        h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        scores = h.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis = 1)

        return y_pred