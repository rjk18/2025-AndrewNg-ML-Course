import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

"""单变量线性回归"""
print("=== 单变量线性回归 ===")

# 导入数据
path = 'input\\1_线性回归\\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

print("数据前5行:")
print(data.head())
print("\n数据统计:")
print(data.describe())

# 数据可视化
plt.figure(figsize=(8, 5))
data.plot(kind='scatter', x='Population', y='Profit', title='人口与利润关系')
plt.savefig('output\\1_线性回归\\1_1_scatter_plot.png')
#plt.close()

# 数据预处理
data.insert(0, 'Ones', 1)
cols = data.shape[1]
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

plt.figure(figsize=(6, 4))
plt.plot(x, f, 'r', label='预测线')
plt.scatter(data['Population'], data.Profit, label='训练数据')
plt.legend(loc=2)
plt.xlabel('人口')
plt.ylabel('利润')
plt.title('预测利润 vs 人口规模')
plt.savefig('output\\1_线性回归\\1_1_linear_fit.png')
plt.close()

# 绘制代价下降曲线
plt.figure(figsize=(8, 4))
plt.plot(np.arange(epoch), cost, 'r')
plt.xlabel('迭代次数')
plt.ylabel('代价')
plt.title('误差 vs 训练轮次')
plt.savefig('output\\1_线性回归\\1_1_cost_curve.png')
plt.close()