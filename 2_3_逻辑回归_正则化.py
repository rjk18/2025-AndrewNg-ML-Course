import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def sigmoid(z):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-z))

def cost(theta, X, Y):
    """计算逻辑回归的代价函数"""
    theta = theta.reshape(1, -1) if theta.ndim == 1 else theta
    Y = Y.reshape(1, -1) if Y.ndim == 1 else Y
    # 计算预测值
    z = X @ theta.T
    h = sigmoid(z)

    # 计算损失
    first = Y * np.log(h)  # 逐元素乘法
    second = (1 - Y) * np.log(1 - h)
    return -np.mean(first + second)

def regularized_cost(theta, X, Y, l=1):
    theta_1n = theta[1:]
    regularized_term = l / (2 * len(X)) * np.power(theta_1n, 2).sum()
    return cost(theta, X, Y) + regularized_term


def feature_mapping(x, y, power, as_ndarray=False):
    data = {'f{0}{1}'.format(i-p, p): np.power(x, i-p) * np.power(y, p)
                for i in range(0, power+1)
                for p in range(0, i+1)
           }
    if as_ndarray:
        return pd.DataFrame(data).values
    else:
        return pd.DataFrame(data)

def gradient(theta, X, Y):
    return (1/len(X) * X.T @ (sigmoid(X @ theta.T) - Y))


def regularized_gradient(theta, X, Y, l=1):
    theta_1n = theta[1:]
    regularized_theta = l / len(X) * theta_1n
    #创建一个新的向量，第一个元素为0（对应偏置项）其余部分是计算出的正则化项
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, Y) + regularized_term

def predict(theta, X):
    probability = sigmoid(X @ theta.T)
    return probability >= 0.5

def plot_decision_boundary(theta, X, Y, x1, x2, positive, negative, power=6):
    """绘制决策边界"""
    # 生成网格点
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))

    # 计算每个网格点的预测值
    for i in range(len(u)):
        for j in range(len(v)):
            # 创建特征映射
            mapped_features = feature_mapping(np.array([u[i]]),
                                              np.array([v[j]]),
                                              power=power,
                                              as_ndarray=True)
            # 确保特征维度正确
            if mapped_features.shape[1] != theta.shape[0]:
                # 如果维度不匹配，可能需要调整
                mapped_features = mapped_features[:, :theta.shape[0]]
            # 修改这里：确保转换为标量
            z_value = mapped_features @ theta
            # 使用.item()提取标量值
            z[i, j] = z_value.item() if hasattr(z_value, 'item') else float(z_value)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    # 绘制决策边界
    ax.contour(u, v, z.T, levels=[0], colors='green', linewidths=2)
    # 绘制数据点
    ax.scatter(positive['Test1'], positive['Test2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test1'], negative['Test2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test1 Score')
    ax.set_ylabel('Test2 Score')
    ax.set_title('正则化逻辑回归决策边界 (λ=1)')
    return fig, ax


path = 'input\\2_逻辑回归\\ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Test1', 'Test2', 'Accepted'])

'''
print("数据前5行:")
print( data.head() )
print("\n数据统计:")
print( data.describe() )
'''

# 分离正负样本
positive = data[data['Accepted'] == 1]
negative = data[data['Accepted'] == 0]

# 可视化原始数据
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.scatter(positive['Test1'], positive['Test2'], s=50, c='b', marker='o', label='Accepted')
ax1.scatter(negative['Test1'], negative['Test2'], s=50, c='r', marker='x', label='Rejected')
ax1.legend()
ax1.set_xlabel('Test1 Score')
ax1.set_ylabel('Test2 Score')
ax1.set_title('原始数据分布')
plt.savefig('output\\2_逻辑回归\\2_3_原始数据分布.png')
plt.close()

# 准备数据
x1 = data['Test1'].values
x2 = data['Test2'].values
Y = data['Accepted'].values

# 特征映射
print("\n进行特征映射...")
ProcessedData = feature_mapping(x1, x2, power=6, as_ndarray=False)
print("特征映射后的数据前5行:")
#print(ProcessedData.head())
print(f"特征映射后的数据维度: {ProcessedData.shape}")

# 初始化参数
X = feature_mapping(x1, x2, power=6, as_ndarray=True)
theta = np.zeros(X.shape[1])
print(f"X维度: {X.shape},theta维度: {theta.shape}, y维度: {Y.shape}")

# 计算初始代价
initial_cost = cost(theta, X, Y)
#print(f"\n初始代价 (未正则化): {initial_cost}")

initial_regularized_cost = regularized_cost(theta, X, Y, l=1)
#print(f"初始代价 (正则化, λ=1): {initial_regularized_cost}")

gradientvalue = gradient(theta, X, Y)
#print(f"\n梯度下降 (未正则化): {gradientvalue}")

regularizedvalue = regularized_gradient(theta, X, Y)
#print(f"\n梯度下降 (正则化, λ=1): {regularizedvalue}")

result = opt.minimize(fun=regularized_cost, x0=theta, args=(X, Y), method='TNC', jac=regularized_gradient)
#print(f"回归结果2: {result}")

Y_pred = predict(result.x, X)
print(classification_report(Y, Y_pred))
# 计算准确率
accuracy = np.mean(Y_pred == Y) * 100
print(f"训练集准确率: {accuracy:.2f}%")

# 绘制决策边界
print("\n绘制决策边界...")
fig2, ax2 = plot_decision_boundary(result.x, X, Y, x1, x2, positive, negative)
plt.savefig('output\\2_逻辑回归\\2_3_决策边界.png')
plt.show()