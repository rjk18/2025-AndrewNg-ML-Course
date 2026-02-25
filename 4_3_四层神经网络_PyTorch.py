import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import os

"""
手写数字识别 - PyTorch 实现（四层神经网络）

本代码使用PyTorch框架实现手写数字识别。
使用MNIST数据集（28x28像素），通过PyTorch的DataLoader进行数据加载和批处理。

PyTorch特有概念说明：
1. torch.nn.Module: 神经网络模块的基类，所有网络都应继承此类
2. DataLoader: 数据加载器，支持批量加载、打乱、多进程等
3. transforms: 数据预处理转换管道
4. torch.optim: 优化器模块
5. torch.nn.functional: 包含各种神经网络函数的模块

网络结构：
输入层 (784) → 隐藏层1 (64) → 隐藏层2 (64) → 隐藏层3 (64) → 输出层 (10)
激活函数：ReLU（隐藏层），log_softmax（输出层）
损失函数：负对数似然损失（NLL Loss）

与TensorFlow版本的主要区别：
1. 手动定义前向传播过程（forward方法）
2. 显式调用loss.backward()进行反向传播
3. 手动更新优化器（optimizer.step()）
4. 需要显式清零梯度（net.zero_grad()）
5. 使用log_softmax + NLL Loss等价于交叉熵损失

训练流程：
1. 初始化网络和优化器
2. 前向传播计算输出
3. 计算损失
4. 反向传播计算梯度
5. 优化器更新权重
6. 重复2-5直至训练完成

数据集：
- MNIST数据集（手写数字，28x28灰度图，6万训练+1万测试）
- 自动下载到当前目录

注意：本代码仅训练2个epoch作为演示，实际应用可能需要更多epoch。
"""

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    for n, (x, _) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))

        # 创建保存目录（如果不存在）
        os.makedirs('output/4_神经网络', exist_ok=True)

        # 生成唯一文件名（使用循环索引n）
        file_path = f'output/4_神经网络/4_3_手写数字识别结果_{n}.png'

        plt.figure()
        plt.imshow(x[0].view(28, 28), cmap='gray')  # 添加灰度图配色
        plt.title(f"Prediction: {int(predict)}")
        plt.savefig(file_path)
        plt.close()  # 关闭图形避免内存泄漏

if __name__ == "__main__":
    main()
