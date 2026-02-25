import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
import seaborn as sns

"""
多项式回归模型：偏差-方差权衡分析

本代码演示了如何使用多项式回归解决非线性问题，并通过学习曲线、正则化和交叉验证
来分析模型的偏差-方差权衡。代码实现了岭回归（Ridge Regression）来防止过拟合。

主要功能：
1. 加载和可视化原始数据
2. 实现线性回归和多项式回归
3. 绘制学习曲线诊断偏差/方差问题
4. 使用特征工程（多项式扩展和标准化）
5. 通过正则化（L2惩罚）控制模型复杂度
6. 使用验证集选择最优正则化参数λ

数据说明：
- 数据来自ex5data1.mat，包含：
  X: 水库水位（特征）
  y: 大坝流出水量（目标）
  Xval, yval: 验证集
  Xtest, ytest: 测试集

关键概念：
- 偏差（Bias）：模型对训练数据的拟合不足
- 方差（Variance）：模型对训练数据的过拟合
- 正则化：通过惩罚大参数值来防止过拟合
- 学习曲线：展示训练误差和验证误差随样本数变化
"""


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 数据加载 =================

def load_data(path):
    d = sio.loadmat(path)
    # 使用 np.ravel 展平，确保 y 是 (m,) 而不是 (m, 1)，避免后续矩阵运算歧义
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])


path = "input\\5_偏差方差\\ex5data1.mat"
X, y, Xval, yval, Xtest, ytest = load_data(path)

# 可视化原始数据
df = pd.DataFrame({'水库中的水位': X, '大坝中流出的水量': y})
sns.lmplot(x='水库中的水位', y='大坝中流出的水量', data=df, fit_reg=False)
plt.title("原始数据")
plt.tight_layout()  # 自动调整布局
plt.savefig('output\\5_偏差方差\\5_1_原始数据.png')
plt.show()

# 为线性回归添加截距项
def add_intercept(X):
    return np.insert(X.reshape(X.shape[0], -1), 0, 1, axis=1)

X_inter = add_intercept(X)
Xval_inter = add_intercept(Xval)
Xtest_inter = add_intercept(Xtest)

#print(f"\nX_inter维度: {X_inter.shape}, Xval_inter维度: {Xval_inter.shape}, Xtest_inter维度: {Xtest_inter.shape}")


# ================= 2. 核心算法逻辑 =================

def cost(theta, X, y):
    """计算基础代价 (MSE) - 均方误差"""
    m = X.shape[0]          # 样本数量
    inner = X @ theta - y   # 预测值 - 真实值
    return (inner.T @ inner) / (2 * m)  # MSE公式

def regularized_cost(theta, X, y, l=1):
    """带L2正则化的代价函数（Ridge回归）"""
    m = X.shape[0]
    reg_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()
    return cost(theta, X, y) + reg_term

def regularized_gradient(theta, X, y, l=1):
    """带正则化的梯度计算"""
    m = X.shape[0]
    grad = (X.T @ (X @ theta - y)) / m
    reg_term = (l / m) * theta
    reg_term[0] = 0  # 截距项不参与正则化
    return grad + reg_term

def train_model(X, y, l=1):
    """训练模型：使用 TNC 优化算法"""
    theta = np.zeros(X.shape[1])        # 初始化参数为零向量
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient)
    return res.x     # 返回最优参数


# ================= 3. 学习曲线诊断 =================

def plot_learning_curve(X, y, Xval, yval, l):
    """绘制学习曲线，用于诊断偏差/方差问题"""
    m = X.shape[0]
    training_cost, cv_cost = [], []     # 存储训练误差和验证误差

    for i in range(1, m + 1):
        # 训练模型时使用前 i 个样本
        res_theta = train_model(X[:i, :], y[:i], l)
        # 计算代价时不带正则化项，这样比较才有意义
        training_cost.append(cost(res_theta, X[:i, :], y[:i]))
        cv_cost.append(cost(res_theta, Xval, yval))

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, m + 1), training_cost, label='训练误差')
    plt.plot(np.arange(1, m + 1), cv_cost, label='验证误差')
    plt.title(f'学习曲线 (lambda = {l})')
    plt.xlabel('训练样本数量')
    plt.ylabel('误差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # 自动调整布局
    plt.savefig('output\\5_偏差方差\\5_1_学习曲线.png')
    plt.show()

# 线性回归的学习曲线 (典型的高偏差/欠拟合形态)
plot_learning_curve(X_inter, y, Xval_inter, yval, 0)


# ================= 4. 多项式回归与特征工程 =================

def poly_features(x, power):
    """生成多项式特征"""
    x = x.reshape(-1, 1)
    for i in range(2, power + 1):
        x = np.insert(x, x.shape[1], np.power(x[:, 0], i), axis=1)
    return x


def normalize_features(X, mean=None, std=None):
    """
    特征归一化：支持传入预设的均值和标准差
    这对保证验证集/测试集与训练集遵循同一缩放尺度至关重要
    """
    if mean is None or std is None:
        mean = np.mean(X, axis=0)       # 计算均值
        std = np.std(X, axis=0)         # 计算标准差

    std[std == 0] = 1                   # 防止除零错误
    X_norm = (X - mean) / std           # 标准化公式
    return X_norm, mean, std

# 准备 8 次多项式数据
power = 8
X_poly = poly_features(X, power)            # 原始训练集 -> 多项式特征
X_poly_norm, train_mean, train_std = normalize_features(X_poly)
# print(f"\nX_poly维度: {X_poly.shape}")
# print(f"\nX_poly_norm维度: {X_poly_norm.shape}, train_mean维度: {train_mean.shape}, train_std维度: {train_std.shape}")
X_poly_final = add_intercept(X_poly_norm)

# 验证集和测试集必须使用训练集的 mean 和 std
Xval_poly = poly_features(Xval, power)
Xval_poly_norm, _, _ = normalize_features(Xval_poly, train_mean, train_std)
Xval_poly_final = add_intercept(Xval_poly_norm)

Xtest_poly = poly_features(Xtest, power)
Xtest_poly_norm, _, _ = normalize_features(Xtest_poly, train_mean, train_std)
Xtest_poly_final = add_intercept(Xtest_poly_norm)


# ================= 5. 可视化多项式拟合曲线 =================

def plot_poly_fit(l=0):
    theta_poly = train_model(X_poly_final, y, l)

    # 绘制散点
    plt.scatter(X, y, marker='x', c='r', label='训练数据')

    # 生成密集的点来绘制平滑曲线
    x_range = np.linspace(X.min() - 15, X.max() + 15, 100)
    x_poly_range = poly_features(x_range, power)
    x_poly_norm_range, _, _ = normalize_features(x_poly_range, train_mean, train_std)
    x_poly_final_range = add_intercept(x_poly_norm_range)

    plt.plot(x_range, x_poly_final_range @ theta_poly, '--', label='多项式拟合')
    plt.title(f'多项式回归拟合 (lambda = {l})')
    plt.xlabel('水库中的水位')
    plt.ylabel('大坝中流出的水量')
    plt.legend()
    plt.savefig('output\\5_偏差方差\\5_1_多项式回归拟合.png')
    plt.show()

# 展示 lambda=0 时的过拟合
plot_poly_fit(l=0)
# 展示 lambda=0 时的学习曲线 (典型的高方差/过拟合形态)
plot_learning_curve(X_poly_final, y, Xval_poly_final, yval, 0)


# ================= 6. 自动寻找最优 Lambda =================

l_candidates = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
train_err, val_err = [], []

for l in l_candidates:
    curr_theta = train_model(X_poly_final, y, l)                # 用当前λ训练模型
    train_err.append(cost(curr_theta, X_poly_final, y))         # 训练误差
    val_err.append(cost(curr_theta, Xval_poly_final, yval))     # 验证误差

plt.plot(l_candidates, train_err, label='训练误差')
plt.plot(l_candidates, val_err, label='交叉验证')
plt.xlabel('lambda')
plt.ylabel('误差')
plt.legend()
plt.title('使用验证集选择 lambda')
plt.savefig('output\\5_偏差方差\\5_1_使用验证集选择lambda.png')
plt.show()

best_l = l_candidates[np.argmin(val_err)]
print(f"最优 Lambda 为: {best_l}")

# 最终在测试集上评估
final_theta = train_model(X_poly_final, y, best_l)
print(f"测试集误差 (lambda={best_l}): {cost(final_theta, Xtest_poly_final, ytest):.4f}")

