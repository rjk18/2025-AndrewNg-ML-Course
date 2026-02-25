import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, Y):
    # 确保theta是二维的以便转置
    theta = theta.reshape(1, -1) if theta.ndim == 1 else theta

    # 计算预测值
    z = X @ theta.T
    h = sigmoid(z)

    # 计算损失
    first = Y * np.log(h)  # 逐元素乘法
    second = (1 - Y) * np.log(1 - h)
    return -np.mean(first + second)

# 计算步长
def gradient(theta, X, Y):
    # 确保theta是二维的以便转置
    theta = theta.reshape(1, -1) if theta.ndim == 1 else theta

    return (1/len(X) * X.T @ (sigmoid(X @ theta.T) - Y))

def predict(theta, X):
    probability = sigmoid(X @ theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


path = 'input\\2_逻辑回归\\ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
"""
print("数据前5行:")
print( data.head() )
print("\n数据统计:")
print( data.describe() )
"""

positive = data[ data['Admitted'].isin([1]) ]
negative = data[ data['Admitted'].isin([0]) ]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
plt.savefig('output\\2_逻辑回归\\2_1_未处理.png')
plt.close()
#plt.show()

nums = np.arange(-10, 10, step=0.5)
fig1, ax = plt.subplots(figsize=(12, 8))
ax.plot(nums, sigmoid(nums), 'r')
plt.savefig('output\\2_逻辑回归\\2_1_sigmoid.png')
plt.close()
#plt.show()

# 数据预处理
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1].values  # 使用.values获取numpy数组
Y = data.iloc[:, cols - 1:cols].values
theta = np.array([0, 0, 0])  # 使用一维数组
print(f"\nX维度: {X.shape}, theta维度: {theta.shape}, y维度: {Y.shape}")

# 计算初始的代价
cost_val = cost(theta.reshape(1, -1), X, Y)
print(f"初始代价: {cost_val}")
gradient_val = gradient(theta.reshape(1, -1), X, Y)
print(f"梯度下降: {gradient_val}")

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
print(f"回归结果: {result}")

optimal_theta = result[0]
cost_val_1 = cost(optimal_theta, X, Y)
print(f"代价: {cost_val_1}")

theta_min = np.array(optimal_theta).reshape(1, -1)
predictions = predict(theta_min, X)
correct = [1 if a^b == 0 else 0 for (a,b) in zip(predictions, Y)]
accuracy = (sum(correct) / len(correct))
print('accuracy = {0:.2f}%'.format(accuracy*100))

print(classification_report(Y, predictions))

coef_intercept = -optimal_theta[0] / optimal_theta[2]
coef_slope = -optimal_theta[1] / optimal_theta[2]

x = np.arange(30, 100, 0.5)
y = coef_intercept + coef_slope * x

fig2, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.plot(x, y, label='Decision Boundary', c='grey')
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
plt.savefig('output\\2_逻辑回归\\2_1_决策边界.png')
plt.show()











