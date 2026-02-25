import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.metrics import classification_report

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(path, transpose=True):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    y = y.reshape(y.shape[0])
    print(type(X))
    if transpose:
        X = np.array([im.reshape((20,20)).T.reshape(400) for im in X])
    return X, y

def plot_an_image(image):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20,20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.savefig('output\\3_神经网络\\3_1_手写数字.png')
    #plt.show()
    plt.close()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    first = y * np.log(sigmoid(X @ theta.T))
    second = (1 - y) * np.log(1 - sigmoid(X @ theta.T))
    return -np.mean(first + second)

def regularized_cost(theta, X, y, l):
    reg = l / (2 * len(X)) * (theta[1:] ** 2).sum()
    return cost(theta, X, y) + reg

def gradient(theta, X, y, l):
    error = sigmoid(X@theta.T) - y
    grad = X.T @ error / len(X)
    reg = theta * l / len(X)
    reg[0] = 0
    return grad + reg

def logistic_regression(X, y, l=1):
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun = regularized_cost, x0=theta, args=(X, y, l), method='TNC', jac=gradient, options={'disp': True})
    return res.x

def predict(theta, X):
    prob = sigmoid(X @ theta)
    return [1 if i >= 0.5 else 0 for i in prob]

raw_x, raw_y = load_data('input\\3_神经网络\\ex3data1.mat')
print(f"raw_x维度: {raw_x.shape},raw_y维度: {raw_y.shape}")

pick_one = np.random.randint(0, 5000)
plot_an_image(raw_x[pick_one, :])
print('this should be {}'.format(raw_y[pick_one]))

X = np.insert(raw_x, 0, np.ones(raw_x.shape[0]), axis=1)
print(f"X维度: {X.shape}")

y = []
for k in range(1, 11):
    y.append([1 if i==k else 0 for i in raw_y])
y = np.array([y[-1]] + y[:-1])
print(f"y维度: {y.shape}")
print(f"y0维度: {y[0].shape}")

theta_0 = logistic_regression(X, y[0])
print(f"theta_0维度: {theta_0.shape}")

y_pred = predict(theta_0, X)
print('Accurary = {}'.format(np.mean(y[0] == y_pred)))

theta_k = np.array([logistic_regression(X, y[k]) for k in range(10)])
print(f"theta_k维度: {theta_k.shape}")

prob_matrix = sigmoid(X @ theta_k.T)    # 计算概率矩阵
np.set_printoptions(suppress=True)      # 抑制科学计数法显示

y_pred = np.argmax(prob_matrix, axis=1) # 返回每行最大的列索引
y_pred = np.array([10 if i == 0 else i for i in y_pred])
print(y_pred)

print(classification_report(raw_y, y_pred))

