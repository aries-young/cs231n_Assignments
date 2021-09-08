import numpy as np

class KNearestNeighbor():
    """ a KNN classifier with L2 distance """
    
    def __init__(self):
        pass
    
    def train(self, X, y):
        """
        对于 k-nearest neighbors 训练的过程仅是记住训练数据

        输入：
        - X：一个大小为 (num_train, D) 的数组存储训练数据，有 num_train 个样本，每个样本 D 维
        - y：一个大小为 (N,) 的数组存储标签
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k = 1, num_loops = 0):
        """
        预测测试数据的标签

        输入：
        - X：一个大小为 (num_train, D) 的数组存储测试数据，有 num_train 个样本，每个样本 D 维
        - k：最近邻的个数
        - num_loops：用于决定使用哪种方法计算训练点和测试点之间的距离

        返回：
        - y：一个大小为 (num_test,) 的数组存储测试标签
        """
        if num_loops == 0:
            dists = self.compute_dis_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_dis_one_loops(X)
        elif num_loops == 2:
            dists = self.compute_dis_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k = k)
    
    def compute_dis_two_loops(self, X):
        """
        在训练数据和测试数据之间使用一个嵌套循环计算每个测试点(X)和每个训练点(X_train)之间的距离

        输入：
        - X: 一个大小为 (num_test, D) 的数组存储测试数据

        返回：
        - dists：一个大小为 (num_test, num_train) 的数组，[i, j] 表示第 i 个测试点和第 j 个训练
        点的欧氏距离
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                # 计算 L2 距离
                dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
        return dists

    def compute_dis_one_loops(self, X):
        """
        使用一层循环来计算每个测试点和训练点之间的距离
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis = 1))
        return dists

    def compute_dis_no_loops(self, X):
        """
        不显示使用循环来计算每个测试点和训练点之间的距离
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # 利用矩阵操作
        dists = np.sqrt(-2 * np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis = 1) + np.transpose([np.sum(np.square(X),axis = 1)]))
        return dists 
    
    def predict_labels(self, dists, k = 1):
        """
        输入一个测试点和训练点的距离矩阵用于为每个测试点标注标签

        输入：
        - dists：距离矩阵

        返回：
        - y：一个大小为 (num_test,) 的数据存储着测试数据的标签
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            closest_y = [] # 数组大小为 k，用于存储距离该测试点最近的 k 个邻居
            # 用距离矩阵去找到 k 个最近邻邻居
            # 用 y_train 来标注这 k 个邻居
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            # 从 closet_y 中选出最可能属于的一个标签，给第 i 个测试树打上标签
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred    
            
