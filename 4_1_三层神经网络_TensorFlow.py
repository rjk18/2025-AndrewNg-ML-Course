import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

"""
手写数字识别 - TensorFlow 实现（三层神经网络）

本代码实现了一个400-25-10结构的三层全连接神经网络，用于识别20x20像素的手写数字。
主要步骤包括：数据加载、预处理、模型构建、训练、评估和可视化。

作者：[一枝一卒]
日期：[2026]
版本：1.0

功能概述：
1. 加载MATLAB格式的手写数字数据集
2. 数据预处理（标签转换、One-Hot编码、数据集划分）
3. 构建三层神经网络（输入层400节点，隐藏层25节点，输出层10节点）
4. 使用Adam优化器和交叉熵损失函数训练模型
5. 评估模型性能并可视化结果

数据说明：
- 输入：400维向量（20x20像素图像展平）
- 输出：10维向量（对应数字0-9的概率）
- 标签：数字1-9对应标签1-9，数字0对应标签10（需要转换为0）

文件结构：
- input\4_神经网络\ex4data1.mat: 原始数据文件
- handwritten_digit_model.h5: 保存的模型文件（可选）

运行环境：
- Python 3.7+
- TensorFlow 2.x
- scikit-learn, matplotlib, numpy, scipy
"""

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
print("正在加载数据...")
path =  "input\\4_神经网络\\ex4data1.mat"
data = sio.loadmat(path)
raw_x = data['X']
raw_y = data['y']

print(f"数据形状: X={raw_x.shape}, y={raw_y.shape}")

# 2. 数据预处理
# 将标签10转换为0（因为数字0的标签是10）
raw_y = np.where(raw_y == 10, 0, raw_y).reshape(-1, 1)

# One-Hot编码
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(raw_y)
print(f"One-Hot编码后y的形状: {y_onehot.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(   raw_x, y_onehot, test_size=0.2, random_state=42)
print(f"X训练集: {X_train.shape}, X测试集: {X_test.shape},Y训练集: {y_train.shape}, Y测试集: {y_test.shape},")

# 3. 构建三层神经网络模型（400-25-10）
print("\n构建三层神经网络模型...")
model = tf.keras.Sequential()

# 输入层（400个特征） + 隐藏层（25个神经元）
model.add(tf.keras.layers.Dense(
    units=25,  # 隐藏层神经元数量
    activation='sigmoid',  # 激活函数
    input_shape=(400,),  # 输入特征维度
    name='hidden_layer'
))

# 输出层（10个神经元，对应0-9数字）
model.add(tf.keras.layers.Dense(
    units=10,  # 输出层神经元数量
    activation='softmax',  # 多分类使用softmax
    name='output_layer'
))

# 4. 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 优化器
    loss='categorical_crossentropy',  # 损失函数
    metrics=['accuracy']  # 评估指标
)

# 5. 显示模型结构
print("\n模型结构:")
model.summary()

# 6. 训练模型
print("\n开始训练模型...")
history = model.fit(
    X_train, y_train,
    epochs=100,  # 训练轮数
    batch_size=32,  # 批次大小
    validation_split=0.2,  # 验证集比例
    verbose=1  # 显示训练进度
)

# 7. 评估模型
print("\n评估模型性能...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")

# 8. 进行预测
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)  # 获取预测的类别
y_true = np.argmax(y_test, axis=1)  # 获取真实的类别

# 计算准确率
accuracy = np.mean(y_pred == y_true)
print(f"\n手动计算的准确率: {accuracy:.4%}")


# 9. 显示一些预测结果
def show_predictions(X, y_true, y_pred, num_samples=5):
    """显示预测结果"""
    indices = np.random.choice(len(X), num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i, idx in enumerate(indices):
        # 重塑图像为20x20
        img = X[idx].reshape(20, 20).T
        true_label = y_true[idx]
        pred_label = y_pred[idx]

        axes[i].imshow(img, cmap='gray')
        title_color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=title_color)
        axes[i].axis('off')

    plt.suptitle('手写数字识别结果', fontsize=16)
    plt.tight_layout()
    plt.savefig('output\\4_神经网络\\4_1_手写数字识别结果.png')
    plt.show()


print("\n显示部分预测结果:")
show_predictions(X_test, y_true, y_pred, num_samples=5)


# 10. 可视化训练过程
def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    ax1.plot(history.history['loss'], label='训练损失')
    ax1.plot(history.history['val_loss'], label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)

    # 绘制准确率曲线
    ax2.plot(history.history['accuracy'], label='训练准确率')
    ax2.plot(history.history['val_accuracy'], label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('训练和验证准确率')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('训练过程可视化', fontsize=14)
    plt.tight_layout()
    plt.savefig('output\\4_神经网络\\4_1_训练过程可视化.png')
    plt.show()


plot_training_history(history)

# 11. 混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.savefig('output\\4_神经网络\\4_1_混淆矩阵.png')
plt.show()

# 12. 分类报告
print("\n详细分类报告:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

# 13. 保存模型
#model.save('handwritten_digit_model.h5')
#print("\n模型已保存为: handwritten_digit_model.h5")

# 14. 加载模型（示例）
# loaded_model = tf.keras.models.load_model('handwritten_digit_model.h5')
# print("模型加载成功！")

print("\n" + "=" * 60)
print("模型训练完成！")
print(f"最终测试准确率: {test_accuracy:.2%}")
print("=" * 60)