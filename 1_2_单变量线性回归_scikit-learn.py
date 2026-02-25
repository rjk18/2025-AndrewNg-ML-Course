import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入数据
path = 'input\\1_线性回归\\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 准备数据
X = data['Population'].values.reshape(-1, 1)  # scikit-learn期望二维特征
y = data['Profit'].values

# 使用scikit-learn训练模型
model = linear_model.LinearRegression(fit_intercept=True)  # 默认包含截距项
model.fit(X, y)

# 打印模型参数
print(f"scikit-learn模型参数:")
print(f"截距 (theta0): {model.intercept_:.4f}")
print(f"系数 (theta1): {model.coef_[0]:.4f}")

# 生成预测回归线
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(X_line)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='训练数据', color='blue')
plt.plot(X_line, y_pred, 'r-', linewidth=2, label='scikit-learn预测线')

# 添加标题和标签
plt.xlabel('人口 (单位: 10,000s)', fontsize=12)
plt.ylabel('利润 (单位: $10,000)', fontsize=12)
plt.title('线性回归: 利润 vs 人口规模 (scikit-learn实现)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')

# 保存和显示
plt.tight_layout()
plt.savefig('output\\1_线性回归\\1_2_sklearn_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n图形已保存为 'sklearn_comparison.png'")

# 计算并显示评估指标
y_pred_all = model.predict(X)
mse = np.mean((y_pred_all - y) ** 2)
r2 = model.score(X, y)

print(f"\n模型性能:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 分数: {r2:.4f}")
