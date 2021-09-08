import numpy as np

# 在该文件实现了各种常用于训练神经网络的一阶更新的规则
# 每种更新规则都输入当前的权重和损失对应当前权重的梯度，然后结算下一次的权重
# 每种根性规则的接口是相同的
# 
# def update(w, dw, config = None):
#
# 输入：
# - w：权重
# - dw：损失对应 w 的梯度
# -config：存储超参数的字典。如果更新规则要求在多次迭代间暂存信息的话，config 将会暂存这些信息
#
# 返回：
# - next_w：更新后的下个点
# - config：用于将暂存的信息传入下一次迭代中
#
# 注意：对大部分的更新规则来说，默认的学习率是表现不好的。但是，对于其他的超参数，默认值表现还是不错的
# 从效率的角度考虑，更新规则采用本地更新，不引入额外的临时变量，即 next_w = w

def sgd(w, dw, config = None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config

def sgd_momentum(w, dw, config = None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    config['velocity'] = v

    return next_w, config


def rmsprop(x, dx, config = None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    # 使用 RMSprop 更新公式
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * (dx ** 2)
    next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])

    return next_x, config

def adam(x, dx, config = None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    next_x = None
    # 使用 Adam 更新公式
    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dx ** 2)
    mb = config['m'] / (1 - config['beta1'] ** config['t'])
    vb = config['v'] / (1 - config['beta2'] ** config['t'])
    next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])

    return next_x, config