# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.cifar10 import load_cifar10
from model.lenet5 import LeNet5
import matplotlib.pyplot as plt2
from common.optimizer import *
import matplotlib.pyplot as plt

# for reproducibility
np.random.seed(0)

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_cifar10(normalize=True, flatten=False, one_hot_label=True)

# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network = LeNet5(input_dim=(3, 32, 32),
                 conv_param_1={'filter_num': 32, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_2={'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_3={'filter_num': 256, 'filter_size': 2, 'pad': 1, 'stride': 1},
                 conv_param_4={'filter_num': 128, 'filter_size': 2, 'pad': 1, 'stride': 1},
                 weight_init_std=0.025)

# optimizer = SGD()
optimizer = Adam(lr=0.001)

iters_num = 30000
train_size = x_train.shape[0]
batch_size = 16
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grads = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grads = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    params = network.params

    # 갱신
    # for key in ('W1', 'b1', 'W2', 'b2'):
    #     network.params[key] -= learning_rate * grad[key]
    optimizer.update(params, grads)
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % (iter_per_epoch/10) == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('iter: (%d:%d)' % (i, iters_num), 'train acc: ', train_acc, ' test acc: ', test_acc)

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
plt.subplot(211)
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Accuracy Plot")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.subplot(212)
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='test acc')
plt.xlabel("Epoch")
plt.ylabel("Testing Loss")

plt.show()

# 파라미터 저장
path_dir = './ckpt'
file_name = "lenet5_params.pkl"
if not os.path.isdir(path_dir):
    os.mkdir(path_dir)

network.save_params(os.path.join(path_dir, file_name))
print("Parameter Save Complete!")