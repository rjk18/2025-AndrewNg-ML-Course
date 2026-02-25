import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class LinearRegression:
    """线性回归实现类"""

    def __init__(self):
        self.theta = None
        self.cost_history = None

    def computeCost(self, X, y, theta):
        """计算代价函数"""
        inner = np.power(((X * theta.T) - y), 2)
        return np.sum(inner) / (2 * len(X))

    def gradientDescent(self, X, y, theta, alpha, epoch):
        """批量梯度下降算法"""
        temp = np.matrix(np.zeros(theta.shape))
        parameters = int(theta.flatten().shape[1])
        cost = np.zeros(epoch)
        m = X.shape[0]

        for i in range(epoch):
            temp = theta - (alpha / m) * (X * theta.T - y).T * X
            theta = temp
            cost[i] = self.computeCost(X, y, theta)

        return theta, cost

    def normalEqn(self, X, y):
        """正规方程求解"""
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        return theta

# 导入数据
path = 'input\\1_线性回归\\ex1data2.txt'
data2 = pd.read_csv(path, names=['Size', 'Bedrooms', 'Price'])

print("数据前5行:")
print(data2.head())

# 特征归一化
data2_norm = (data2 - data2.mean()) / data2.std()
print("\n归一化后数据:")
print(data2_norm.head())

# 数据预处理
data2_norm.insert(0, 'Ones', 1)
cols = data2_norm.shape[1]
X2 = data2_norm.iloc[:, 0:cols - 1]
y2 = data2_norm.iloc[:, cols - 1:cols]

# 转换为矩阵
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

print(f"\nX维度: {X2.shape}, theta维度: {theta2.shape}, y维度: {y2.shape}")

lr = LinearRegression()
initial_cost = lr.computeCost(X2, y2, theta2)
print(f"初始代价: {initial_cost}")

# 梯度下降
alpha = 0.01
epoch = 1000
g2, cost2 = lr.gradientDescent(X2, y2, theta2, alpha, epoch)
final_cost2 = lr.computeCost(X2, y2, g2)

print(f"最终代价: {final_cost2}")
print(f"最终参数: {g2}")

# 绘制代价下降曲线
plt.figure(figsize=(12, 8))
plt.plot(np.arange(epoch), cost2, 'r')
plt.xlabel('迭代次数')
plt.ylabel('代价')
plt.title('误差 vs 训练轮次 (多变量)')
plt.savefig('output\\1_线性回归\\1_4_multi_cost_curve.png')
plt.close()