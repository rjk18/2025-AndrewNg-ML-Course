import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.metrics import classification_report

#path =

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

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


theta1, theta2 = load_weight('input\\3_神经网络\\ex3weights.mat')
print(f"theta1维度: {theta1.shape},theta2维度: {theta2.shape}")

X, y = load_data('input\\3_神经网络\\ex3data1.mat', transpose=False)
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
print(f"X维度: {X.shape},y维度: {y.shape}")

a1 = X
z2 = a1 @ theta1.T
z2 = np.insert(z2, 0, np.ones(z2.shape[0]), axis=1)
print(f"z2维度: {z2.shape}")
a2 = sigmoid(z2)

z3 = a2 @ theta2.T
print(f"z3维度: {z3.shape}")
a3 = sigmoid(z3)

print(f"a3维度: {a3.shape}")

y_pred = np.argmax(a3, axis=1)+1         # 返回每行最大的列索引
print(y_pred)
print(classification_report(y, y_pred))






