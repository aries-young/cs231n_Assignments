import numpy as np
from cs231n import optim

class Solver():
    """
    Solver 封装了训练一个分类器所必要的操作。Solver 使用随机梯度下降，
    不同的更新规则在 optim.py 中

    Solver 输入训练集、验证集和标签集，它可以周期性的检查
    分类器在训练集和验证集上的准确率，以此检查是否发生过拟合

    要训练模型，先要创建一个 Solver 的实例，然后传入模型、数据
    和一些选择参数。然后，就可以调用 train() 进行最优化训练模型

    train() 运行返回后，model.params 将储存模型在验证机上最优
    的超参数配置。另外，solver.loss_history 会存储训练过程中的
    所有损失值，solver.train_acc_history 和 solver.val_acc_history
    会记录模型在训练集和验证集上的准确率

    例如：

    data = {
        'X_train': # training data
        'y_train': # training labels
        'X_val': # validation data
        'X_train': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, 
                    data,
                    update_rule = 'sgd',
                    optim_config={'learning_rate': 1e-3,},
                    lr_decay=0.95,
                    num_epochs=10, 
                    batch_size=100,
                    print_every=100)
    solver.train()

    传入 Solver 的 model 要求有如下的接口：
    
    - model.params：参数字典
    - model.loss(X, y)：损失函数，计算训练时的损失和梯度，以及测试时分类得分
        输入：
        - X：输入数据的 minibatch (N, d_1, ..., d_k)
        - y：标签数组
        返回：
        如果 y 不为空，运行验证时前向传播返回
        - scores：评分数组
        如果 y 为空，运行训练时前向传播和反向传播返回元组、
        - loss：损失值
        - grads：梯度字典  
    """

    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            extra = ','.join('"%s"' % k for k in kwargs.keys())
            raise ValueError(('Unrecognized arguments % s' % extra))

        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_relu "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()
    
    def _reset(self):
        """
        设置一些固定的变量以进行优化，不要手动调用此函数
        """
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        进行一步的梯度更新，在 train() 函数中被调用，不要手动调用该函数
        """
        # 生成 minibatch
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        
        # 计算损失和梯度
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # 更新参数
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples = None, batch_size = 100):
        # 可能要对数据进行二次取样
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # 计算预测结果
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(int(num_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis = 1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(int(num_iterations)):
            self._step()

            # 输出训练损失
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations, self.loss_history[-1]))
            
            # 每个周期结束都要自增周期计算器，同时衰减学习率
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # 在第一个周期、最后一个周期以及每个周期结束时检查训练准确率和验证准确率
            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples = 1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (self.epoch, self.num_epochs, train_acc, val_acc))

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                for k, v in self.model.params.items():
                    self.best_params[k] = v.copy()
            self.model.params = self.best_params
            