import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# 关闭所有之前打开的图形
plt.close('all')

path = 'input\\1_线性回归\\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

'''
print("数据前5行:")
print(data.head())
print("\n数据统计:")
print(data.describe())
'''

# 创建新图形并绘制
fig, ax = plt.subplots(figsize=(8, 5))
data.plot(kind='scatter', x='Population', y='Profit', ax=ax, title='原始数据：人口与利润关系')
# 设置中文坐标轴标签
ax.set_xlabel('人口')
ax.set_ylabel('利润')
plt.savefig('output\\1_线性回归\\1_1_原始数据.png')
plt.show()
# 数据预处理
data.insert(0, 'Ones', 1)       #在数据第一列插入名为'Ones'的列，值全为1，用于表示截距项
cols = data.shape[1]                                #获取数据的总列数
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# 转换为矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0, 0])

print(f"\nX维度: {X.shape}, theta维度: {theta.shape}, y维度: {y.shape}")


# 计算初始代价
lr = LinearRegression()
initial_cost = lr.computeCost(X, y, theta)
print(f"初始代价: {initial_cost}")

# 梯度下降
alpha = 0.01
epoch = 2000
final_theta, cost = lr.gradientDescent(X, y, theta, alpha, epoch)
final_cost = lr.computeCost(X, y, final_theta)
print(f"最终代价: {final_cost}")
print(f"最终参数: {final_theta}")

# 绘制拟合直线
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = final_theta[0, 0] + (final_theta[0, 1] * x)

plt.close('all')
plt.figure(figsize=(8, 5))
plt.plot(x, f, 'r', label='预测线')
plt.scatter(data['Population'], data.Profit, label='训练数据')
plt.legend(loc=2)
plt.xlabel('人口')
plt.ylabel('利润')
plt.title('预测利润 vs 人口规模')

plt.savefig('output\\1_线性回归\\1_1_预测利润_人口规模.png')
plt.show()

# 绘制代价下降曲线
plt.close('all')
plt.figure(figsize=(8, 5))
plt.plot(np.arange(epoch), cost, 'r')
plt.xlabel('迭代次数')
plt.ylabel('代价')
plt.title('误差 vs 训练轮次')

plt.savefig('output\\1_线性回归\\1_1_误差_训练轮次.png')
plt.show()

# 1. 准备网格数据
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# 计算网格中每个点的代价
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.matrix([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = lr.computeCost(X, y, t)

# 注意：由于 contour 函数的特性，需要对 J_vals 进行转置
J_vals = J_vals.T

# 2. 绘制等高线图
plt.close('all')
plt.figure(figsize=(8, 5))
# 绘制等高线，np.logspace 用于让等高线分布更均匀（由密到稀）
levels = np.logspace(-2, 3, 20)
contour = plt.contour(theta0_vals, theta1_vals, J_vals, levels=levels, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8) # 显示等高线数值

# 绘制最终训练出的参数点（红叉）
plt.plot(final_theta[0, 0], final_theta[0, 1], 'rx', markersize=10, linewidth=2, label='最优解')
plt.xlabel('截距 b', fontsize=12)
plt.ylabel('权重 w', fontsize=12)
plt.title('代价函数 J(w, b) 等高线图', fontsize=14)
plt.legend()

plt.savefig('output\\1_线性回归\\1_1_代价函数等高线图.png')
plt.show()

# 3. (可选) 绘制 3D 曲面图，更直观
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
surf = ax.plot_surface(T0, T1, J_vals, cmap='viridis', alpha=0.8)
ax.set_xlabel('截距 b', fontsize=12)
ax.set_ylabel('权重 w', fontsize=12)
ax.set_zlabel(r'J(w, b) ')
ax.set_title('代价函数 3D 曲面')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('output\\1_线性回归\\1_1_代价函数3D曲面.png')
plt.show()
