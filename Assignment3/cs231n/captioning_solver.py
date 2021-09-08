import numpy as np

from cs231n import optim
from cs231n.coco_utils import sample_coco_minibatch

class CaptioningSolver():
    """
    CaotioningSolver 封装了训练图片标注模型所必须的逻辑操作
    CaptioningSolver 采用随机梯度下降算法，不同的更新规则在 optim.py 文件中实现
    """
    def __init__(self, model, data, **kwargs):

        self.model = model
        self.data = data
        
        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()


    def _reset(self):
        # Set up some variables for book-keeping
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
        # Make a minibatch of training data
        minibatch = sample_coco_minibatch(self.data,
                                          batch_size=self.batch_size,
                                          split='train')
        captions, features, urls = minibatch

        # Compute loss and gradient
        loss, grads = self.model.loss(features, captions)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.
        
        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
        on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
        much memory.
        
        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
        classified by the model.
        """
        return 0.0
    
        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        num_train = self.data['train_captions'].shape[0]
        ations_per_epoch = max(num_train // self.batch_size, 1)
        num_ations = self.num_epochs * ations_per_epoch

        for t in range(num_ations):
            self._step()

        # Maybe print training loss
        if self.verbose and t % self.print_every == 0:
            print('(ation %d / %d) loss: %f' % (t + 1, num_ations, self.loss_history[-1]))

        # At the end of every epoch, increment the epoch counter and decay the
        # learning rate.
        epoch_end = (t + 1) % ations_per_epoch == 0
        if epoch_end:
            self.epoch += 1
            for k in self.optim_configs:
                self.optim_configs[k]['learning_rate'] *= self.lr_decay

        # Check train and val accuracy on the first ation, the last
        # ation, and at the end of each epoch.
        # TODO: Implement some logic to check Bleu on validation set periodically

        # At the end of training swap the best params into the model
        # self.model.params = self.best_params