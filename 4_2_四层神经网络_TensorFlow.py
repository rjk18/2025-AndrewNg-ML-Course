import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

"""
手写数字识别 - TensorFlow 实现（四层深度神经网络）

本代码实现了一个400-64-64-64-10结构的四层全连接深度神经网络。
使用ReLU激活函数和稀疏分类交叉熵损失，适合处理未进行One-Hot编码的整数标签。

主要改进点：
1. 更深的网络结构（3个隐藏层，每层64个神经元）
2. 使用ReLU激活函数替代Sigmoid，缓解梯度消失问题
3. 使用sparse_categorical_crossentropy损失，无需手动One-Hot编码
4. 修正了图像方向问题（原始数据需要转置）

网络结构：
输入层 (400) → 隐藏层1 (64, ReLU) → 隐藏层2 (64, ReLU) → 隐藏层3 (64, ReLU) → 输出层 (10, Softmax)

数据预处理说明：
- 原始数据中的图像存储方向需要调整：将400维向量重塑为20x20矩阵 → 转置 → 重新展平
- 标签10转换为0（数字0）

训练参数：
- 优化器：Adam，学习率0.001
- 批次大小：32
- 训练轮数：50
- 验证集比例：20%

注意：如果找不到数据文件，代码会生成随机数据作为演示。
"""

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载并预处理数据
path = "input\\4_神经网络\\ex4data1.mat"        # 数据文件路径
try:
    data = sio.loadmat(path)        # 加载MATLAB格式的数据文件
    X = data['X']                   # 提取特征数据（手写数字图像）
    y = data['y'].flatten()         # 提取标签并展平为一维数组

    # 修正图像方向并归一化
    # 原始数据是20x20像素的图像，但存储方式需要调整
    # 这里将每个样本从400维向量 -> 20x20矩阵 -> 转置 -> 重新展平为400维
    X = np.array([im.reshape((20, 20)).T.reshape(400) for im in X])
    y = np.where(y == 10, 0, y)         # 原始数据是20x20像素的图像，但存储方式需要调整,这里将每个样本从400维向量 -> 20x20矩阵 -> 转置 -> 重新展平为400维
except FileNotFoundError:
    print("未找到数据文件，使用生成的模拟数据...")
    X, y = np.random.rand(5000, 400), np.random.randint(0, 10, 5000)

# 2. # 将数据划分为训练集和测试集，测试集占20%，设置随机种子确保可重复性
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nX_train维度: {X_train.shape}, X_test维度: {X_test.shape}, y_train维度: {y_train.shape}, y_test维度: {y_test.shape}")

# 3. 构建 4 层全连接神经网络 (400 -> 64 -> 64 -> 64 -> 10)
model = tf.keras.Sequential([
    # 输入层由 input_shape 指定，第一个隐藏层 64 神经元
    tf.keras.layers.Dense(64, activation='relu', input_shape=(400,), name='Hidden_1'),
    # 第二个隐藏层
    tf.keras.layers.Dense(64, activation='relu', name='Hidden_2'),
    # 第三个隐藏层
    tf.keras.layers.Dense(64, activation='relu', name='Hidden_3'),
    # 输出层 (10个数字，使用 Softmax)
    tf.keras.layers.Dense(10, activation='softmax', name='Output')
])

# 4. 编译模型
# 使用 Adam 优化器，损失函数使用 sparse_categorical_crossentropy (直接输入整数标签，无需手工 one-hot),评估指标为准确率
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. 训练模型
print("\n开始深度训练...")
history = model.fit(
    X_train, y_train,
    epochs=50,              # 深度网络收敛快，20轮通常足够
    batch_size=32,          # 每批32个样本
    validation_split=0.2,     # 从训练集中拿出%作为验证集
    verbose=1               # 显示训练进度条
)

# 6. 评估与可视化预测
print("\n正在测试集上评估...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集准确率: {acc:.2%}")

# 随机选几个结果看看
predictions = model.predict(X_test)
indices = np.random.choice(len(X_test), 5)

plt.figure(figsize=(12, 3))
for i, idx in enumerate(indices):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[idx].reshape(20, 20), cmap='gray')
    pred_label = np.argmax(predictions[idx])
    true_label = y_test[idx]
    plt.title(f"预测: {pred_label}\n真值: {true_label}", color='green' if pred_label==true_label else 'red')
    plt.axis('off')
plt.tight_layout()
plt.savefig('output\\4_神经网络\\4_2_手写数字识别结果.png')
plt.show()