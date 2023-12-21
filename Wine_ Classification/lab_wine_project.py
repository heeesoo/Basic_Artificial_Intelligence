import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import csv

# A = wine
A = pd.read_csv('C:/2023_python_ai/basicai_fa23/202021295_김희수_7주차과제/wine.csv',sep = ',')
x = A.iloc[0:, 1:].values
y = A.iloc[0:, 0].values

g = sns.pairplot(A, hue="#Wine", palette="husl")
plt.show()
A.info() # 와인 품질 나타내는 그래프 출력

# 2. 선택
# 정규화 방법
x_max = x.max(axis=0)
x_normal = x / x_max

x = x_normal.copy()

# 3. 원 핫 인코딩 (one-hot encoding)
label = []
# 0 : '1'
# 1 : '2'
# 2 : '3'

for name in y:
    if name == 1:
        label.append(0)
    elif name == 2:
        label.append(1)
    elif name == 3:
        label.append(2)

print(label)
print(x)

# one-hot encoding
num = np.unique(label, axis=0)
num = num.shape[0]

encoding = np.eye(num)[label]
print(encoding)

y = np.array(label)
y_hot = encoding.copy()

# 데이터 : 13개
# 클래스 개수 : 3개
dim = 13
nb_classes = 3
print('x shape: ', x.shape, 'y shape: ', y.shape)

w = np.random.normal(size=[dim, nb_classes])
b = np.random.normal(size=[nb_classes])

print('w shape: ', w.shape, 'b shape: ', b.shape)

def cross_entropy_error(predict, target):
    delta = 1e-7
    return -np.mean(target * np.log(predict + delta))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x -np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

hypothesis = softmax(np.dot(x, w) + b)
print('hypothesis: ', hypothesis.shape)

eps = 1e-7

num_epoch = 50000
learning_rate = 100
costs = []

m, n = x.shape

for epoch in range(num_epoch):
    # z = x@w + b or
    z = np.dot(x, w) + b
    # print(z.shape)
    hypothesis = softmax(z)
    # print('hypothesis: ', hypothesis.shape)

    # cost = -np.mean(np.log(hypothesis[np.arange(len(y)), y]))
    # or
    cost = y_hot * np.log(hypothesis + eps)
    cost = -cost.mean()

    if cost < 0.00005:
        break

    # https://towardsdatascience.com/softmax-regression-in-python-multi-class-classification-3cb560d90cb2
    # Calculating the gradient of loss w.r.t w and b.
    w_grad = (1 / m) * np.dot(x.T, (hypothesis - y_hot))
    b_grad = (1 / m) * np.sum(hypothesis - y_hot)

    w = w - learning_rate * w_grad
    b = b - learning_rate * b_grad

    costs.append(cost)

    if epoch % 50 == 0:
        print("{0:2} error = {1:.5f}".format(epoch, cost))

plt.figure(figsize=(10, 7))
plt.plot(costs)
plt.xlabel('Epochs')
plt.ylabel('Costs')
plt.show() # cost 값 그래프로 출력


print('--------------------------------------------------------')
print("{0:2} error = {1:.5f}".format(epoch, cost))

success_cnt = 0
for i in range(15000):
    random_num = np.random.randint(0, len(x)-1)
    x1 = x[random_num]
    t = np.dot(x1, w) + b
    z = softmax(t)
    print('t: ', t, 'z: ', z)
    prediction = np.argmax(z)
    print('prediction: ', prediction)
    print('label: ', y_hot[random_num])
    print('--------------------------------------------------------')
    if prediction == np.argmax(y_hot[random_num]):
        success_cnt += 1

print('--------------------------------------------------------')
print('QUIT')
print('Accuracy: %.2f' %((success_cnt/15000)*100),'%') # 정확도 측정하기
