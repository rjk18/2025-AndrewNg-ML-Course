import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
import seaborn as sns

"""
多项式回归与偏差-方差分析 - 完整实现

本代码系统性地演示了如何通过多项式回归解决非线性回归问题，并通过正则化和交叉验证
来优化模型性能。代码展示了从简单线性回归到复杂多项式回归的完整流程，重点在于
理解偏差-方差权衡并找到最优模型复杂度。

主要学习目标：
1. 识别高偏差（欠拟合）和高方差（过拟合）问题
2. 使用多项式特征扩展处理非线性关系
3. 通过L2正则化（岭回归）控制模型复杂度
4. 使用验证集选择最优正则化参数
5. 可视化分析模型性能

代码结构：
1. 数据加载与可视化
2. 辅助函数定义（代价函数、梯度、训练等）
3. 学习曲线绘制与分析
4. 多项式特征工程
5. 正则化参数选择
6. 模型评估与可视化

作者：[一枝一卒]
日期：[2026]
版本：2.0

文件结构：
input\5_偏差方差\ex5data1.mat  # 原始数据
output\5_偏差方差\              # 输出图像目录
"""


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 数据加载 =================

def load_data(path):
    d = sio.loadmat(path)
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])


path = "input\\5_偏差方差\\ex5data1.mat"
X, y, Xval, yval, Xtest, ytest = load_data(path)

# 可视化原始数据
fig = plt.figure()
df = pd.DataFrame({'水库中的水位': X, '大坝中流出的水量': y})
sns.lmplot(x='水库中的水位', y='大坝中流出的水量', data=df, fit_reg=False)
plt.title("原始数据")
plt.tight_layout()
plt.savefig('output\\5_偏差方差\\5_2_原始数据.png')
plt.show()


# ================= 2. 辅助函数 =================

def add_intercept(X):
    """添加截距项"""
    return np.insert(X.reshape(X.shape[0], -1), 0, 1, axis=1)


def cost(theta, X, y):
    """计算基础代价 (MSE)"""
    m = X.shape[0]
    inner = X @ theta - y
    return (inner.T @ inner) / (2 * m)


def regularized_cost(theta, X, y, l=1):
    """带L2正则化的代价函数"""
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
    """训练模型"""
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient)
    return res.x


# ================= 3. 学习曲线诊断 =================

def plot_learning_curve(X, y, Xval, yval, l, model_type="线性"):
    """绘制学习曲线"""
    m = X.shape[0]
    training_cost, cv_cost = [], []

    for i in range(1, m + 1):
        res_theta = train_model(X[:i, :], y[:i], l)
        training_cost.append(cost(res_theta, X[:i, :], y[:i]))
        cv_cost.append(cost(res_theta, Xval, yval))

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, m + 1), training_cost, label='训练误差')
    plt.plot(np.arange(1, m + 1), cv_cost, label='验证误差')
    plt.title(f'{model_type}回归学习曲线 (lambda = {l})')
    plt.xlabel('训练样本数量')
    plt.ylabel('误差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 动态生成文件名
    if model_type == "线性":
        filename = 'output\\5_偏差方差\\5_2_线性学习曲线.png'
    else:
        filename = f'output\\5_偏差方差\\5_2_多项式学习曲线_lambda_{l}.png'

    plt.savefig(filename)
    plt.show()


# ================= 4. 多项式特征工程 =================

def poly_features(x, power):
    """生成多项式特征"""
    x = x.reshape(-1, 1)
    for i in range(2, power + 1):
        x = np.insert(x, x.shape[1], np.power(x[:, 0], i), axis=1)
    return x


def normalize_features(X, mean=None, std=None):
    """特征归一化"""
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

    std[std == 0] = 1
    X_norm = (X - mean) / std
    return X_norm, mean, std


# ================= 5. 多项式拟合可视化 =================

def plot_poly_fit(X, y, X_poly_final, train_mean, train_std, power, l=0, show_learning_curve=False):
    """
    绘制多项式拟合曲线，可选择是否同时显示学习曲线

    参数:
    - X: 原始特征
    - y: 目标值
    - X_poly_final: 多项式特征（已添加截距）
    - train_mean: 训练集均值（用于标准化）
    - train_std: 训练集标准差（用于标准化）
    - power: 多项式次数
    - l: 正则化参数
    - show_learning_curve: 是否显示学习曲线
    """
    # 训练模型
    theta_poly = train_model(X_poly_final, y, l)

    # 创建图形
    if show_learning_curve:
        # 创建一个包含两个子图的图形
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左图：拟合曲线
        ax1 = axes[0]
        ax1.scatter(X, y, marker='x', c='r', label='训练数据')

        # 生成平滑曲线
        x_range = np.linspace(X.min() - 15, X.max() + 15, 100)
        x_poly_range = poly_features(x_range, power)
        x_poly_norm_range, _, _ = normalize_features(x_poly_range, train_mean, train_std)
        x_poly_final_range = add_intercept(x_poly_norm_range)

        ax1.plot(x_range, x_poly_final_range @ theta_poly, '--', label='多项式拟合')
        ax1.set_title(f'多项式回归拟合 (lambda = {l})')
        ax1.set_xlabel('水库中的水位')
        ax1.set_ylabel('大坝中流出的水量')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：学习曲线
        ax2 = axes[1]
        m = X_poly_final.shape[0]
        training_cost, cv_cost = [], []

        for i in range(1, m + 1):
            res_theta = train_model(X_poly_final[:i, :], y[:i], l)
            training_cost.append(cost(res_theta, X_poly_final[:i, :], y[:i]))
            cv_cost.append(cost(res_theta, Xval_poly_final, yval))

        ax2.plot(np.arange(1, m + 1), training_cost, label='训练误差')
        ax2.plot(np.arange(1, m + 1), cv_cost, label='验证误差')
        ax2.set_title(f'多项式学习曲线 (lambda = {l})')
        ax2.set_xlabel('训练样本数量')
        ax2.set_ylabel('误差')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'output\\5_偏差方差\\5_2_多项式拟合与学习曲线_lambda_{l}.png'

    else:
        # 单图：仅拟合曲线
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, marker='x', c='r', label='训练数据')

        # 生成平滑曲线
        x_range = np.linspace(X.min() - 15, X.max() + 15, 100)
        x_poly_range = poly_features(x_range, power)
        x_poly_norm_range, _, _ = normalize_features(x_poly_range, train_mean, train_std)
        x_poly_final_range = add_intercept(x_poly_norm_range)

        plt.plot(x_range, x_poly_final_range @ theta_poly, '--', label='多项式拟合')
        plt.title(f'多项式回归拟合 (lambda = {l})')
        plt.xlabel('水库中的水位')
        plt.ylabel('大坝中流出的水量')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f'output\\5_偏差方差\\5_2_多项式拟合_lambda_{l}.png'

    plt.savefig(filename)
    plt.show()


# ================= 6. 主程序流程 =================

# 准备线性回归数据
X_inter = add_intercept(X)
Xval_inter = add_intercept(Xval)
Xtest_inter = add_intercept(Xtest)

# 1. 线性回归学习曲线
print("正在绘制线性回归学习曲线...")
plot_learning_curve(X_inter, y, Xval_inter, yval, 0, model_type="线性")

# 2. 准备多项式特征
power = 8
X_poly = poly_features(X, power)
X_poly_norm, train_mean, train_std = normalize_features(X_poly)
X_poly_final = add_intercept(X_poly_norm)

# 验证集多项式特征
Xval_poly = poly_features(Xval, power)
Xval_poly_norm, _, _ = normalize_features(Xval_poly, train_mean, train_std)
Xval_poly_final = add_intercept(Xval_poly_norm)

# 测试集多项式特征
Xtest_poly = poly_features(Xtest, power)
Xtest_poly_norm, _, _ = normalize_features(Xtest_poly, train_mean, train_std)
Xtest_poly_final = add_intercept(Xtest_poly_norm)

# 3. 展示过拟合情况（λ=0）
print("正在绘制多项式回归过拟合情况...")
plot_poly_fit(X, y, X_poly_final, train_mean, train_std, power, l=0, show_learning_curve=True)

# 4. 寻找最优lambda
print("正在寻找最优正则化参数...")
l_candidates = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
train_err, val_err = [], []

for l in l_candidates:
    curr_theta = train_model(X_poly_final, y, l)
    train_err.append(cost(curr_theta, X_poly_final, y))
    val_err.append(cost(curr_theta, Xval_poly_final, yval))

plt.figure(figsize=(10, 6))
plt.plot(l_candidates, train_err, label='训练误差')
plt.plot(l_candidates, val_err, label='交叉验证误差')
plt.xlabel('lambda')
plt.ylabel('误差')
plt.legend()
plt.title('使用验证集选择 lambda')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output\\5_偏差方差\\5_2_使用验证集选择lambda.png')
plt.show()

best_l = l_candidates[np.argmin(val_err)]
print(f"最优 Lambda 为: {best_l}")

# 5. 展示正则化后的拟合效果
print("正在绘制正则化后的多项式拟合...")
plot_poly_fit(X, y, X_poly_final, train_mean, train_std, power, l=best_l, show_learning_curve=False)

# 6. 测试集评估
final_theta = train_model(X_poly_final, y, best_l)
test_error = cost(final_theta, Xtest_poly_final, ytest)
print(f"测试集误差 (lambda={best_l}): {test_error:.4f}")

# 7. 可选：绘制不同lambda的拟合效果对比
print("\n可选：绘制不同lambda的拟合效果对比...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, l in enumerate([0, 0.01, 0.1, 1, 3, 10]):
    if idx >= len(axes):
        break

    theta = train_model(X_poly_final, y, l)

    # 生成平滑曲线
    x_range = np.linspace(X.min() - 15, X.max() + 15, 100)
    x_poly_range = poly_features(x_range, power)
    x_poly_norm_range, _, _ = normalize_features(x_poly_range, train_mean, train_std)
    x_poly_final_range = add_intercept(x_poly_norm_range)

    axes[idx].scatter(X, y, marker='x', c='r', s=30, alpha=0.6)
    axes[idx].plot(x_range, x_poly_final_range @ theta, 'b-', linewidth=2)
    axes[idx].set_title(f'λ = {l}')
    axes[idx].set_xlabel('水库水位')
    axes[idx].set_ylabel('流出水量')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('不同正则化参数λ的多项式拟合效果对比', fontsize=16)
plt.tight_layout()
plt.savefig('output\\5_偏差方差\\5_2_不同lambda拟合效果对比.png')
plt.show()

print("\n所有图像已保存至 output\\5_偏差方差\\ 目录下！")