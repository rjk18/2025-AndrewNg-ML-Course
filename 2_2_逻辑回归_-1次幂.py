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
    theta = theta.reshape(1, -1) if theta.ndim == 1 else theta
    z = X @ theta.T
    h = sigmoid(z)
    first = Y * np.log(h)
    second = (1 - Y) * np.log(1 - h)
    return -np.mean(first + second)

def gradient(theta, X, Y):
    theta = theta.reshape(1, -1) if theta.ndim == 1 else theta
    return (1/len(X) * X.T @ (sigmoid(X @ theta.T) - Y))

def predict(theta, X):
    probability = sigmoid(X @ theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

path = 'input\\2_逻辑回归\\ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])

# 检查数据中是否有0值（避免除零错误）
print("检查数据中是否有0值:")
print(f"Exam1最小值: {data['Exam1'].min()}, Exam2最小值: {data['Exam2'].min()}")

# 添加倒数特征
data['Exam1_inv'] = 1 / data['Exam1']
data['Exam2_inv'] = 1 / data['Exam2']

# 数据可视化
positive = data[data['Admitted'] == 1]
negative = data[data['Admitted'] == 0]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
plt.savefig('output\\2_逻辑回归\\2_2_未处理.png')
plt.close()

# 数据预处理 - 使用倒数特征
data.insert(0, 'Ones', 1)
X = np.column_stack((
    data['Ones'].values,
    data['Exam1_inv'].values,
    data['Exam2_inv'].values
))
Y = data['Admitted'].values.reshape(-1, 1)
theta = np.zeros(3)  # [b, w1, w2]

print(f"\nX维度: {X.shape}, theta维度: {theta.shape}, y维度: {Y.shape}")

# 计算初始代价
cost_val = cost(theta, X, Y)
print(f"初始代价: {cost_val}")

# 优化
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
optimal_theta = result[0]
cost_val_1 = cost(optimal_theta, X, Y)
print(f"优化后代价: {cost_val_1}")

# 预测和评估
theta_min = optimal_theta.reshape(1, -1)
predictions = predict(theta_min, X)
accuracy = np.mean(np.array(predictions).reshape(-1, 1) == Y)
print('accuracy = {0:.2f}%'.format(accuracy * 100))
print(classification_report(Y, predictions))

# 绘制倒数决策边界
x1 = np.linspace(30, 100, 100)
x2 = np.linspace(30, 100, 100)
X1, X2 = np.meshgrid(x1, x2)

# 计算网格点的预测值（使用倒数）
Z = theta_min[0][0] + theta_min[0][1] * (1/X1) + theta_min[0][2] * (1/X2)
Z = sigmoid(Z)

fig2, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.contour(X1, X2, Z, levels=[0.5], colors='grey', linewidths=2)
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
ax.set_xlim(30, 100)
ax.set_ylim(30, 100)
plt.savefig('output\\2_逻辑回归\\2_2_决策边界.png')
plt.show()

# 为了更好地理解倒数特征，可以绘制倒数特征空间的数据分布
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 原始特征空间
ax1.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax1.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax1.set_xlabel('Exam1 Score')
ax1.set_ylabel('Exam2 Score')
ax1.set_title('原始特征空间')
ax1.legend()

# 倒数特征空间
ax2.scatter(positive['Exam1_inv'], positive['Exam2_inv'], s=50, c='b', marker='o', label='Admitted')
ax2.scatter(negative['Exam1_inv'], negative['Exam2_inv'], s=50, c='r', marker='x', label='Not Admitted')
ax2.set_xlabel('1/Exam1 Score')
ax2.set_ylabel('1/Exam2 Score')
ax2.set_title('倒数特征空间')
ax2.legend()

plt.tight_layout()
plt.savefig('output\\2_逻辑回归\\2_2_特征空间对比.png')
plt.show()