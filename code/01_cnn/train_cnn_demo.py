
#1、cnn网络搭建，

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. 检查运行设备（CPU）
device = torch.device("cpu")
print(f"当前使用的计算设备: {device}")

# 2. 准备数据集与数据加载器 (DataLoader)
# 将图像转为张量并做归一化：将像素从 [0, 255] 压缩到 [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  #ToTensor自动除以255，所有值落到0-1之间
    transforms.Normalize((0.5,), (0.5,))  #mean是平均值，std是标准差
    #输出=（输入-mean）/std
    #当输入为0时，输出=（0-0.5）/0.5=-1.0，所以最后所有值落在（-1，1）的区间上
])

print("正在自动下载/加载 Fashion-MNIST 数据集（仅几十MB）...")
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

# 每个批次取 64 张图片
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)  #训练集的内容打乱即可，防止模型认为后面要学的都是前面那些东西
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)   #测试集不需要打乱，

# 10 个分类标签名称
classes = [
    'T恤', '裤子', '套头衫', '连衣裙', '外套',
    '凉鞋', '衬衫', '运动鞋', '包', '短靴'
]


# 3. 构建标准卷积神经网络 (CNN)
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # 阶段 1：特征提取 (卷积 + 激活 + 池化)（第一层卷积）
        # 输入单通道灰度图 [1, 28, 28]，输出 16 个特征通道 [16, 28, 28] -> 池化后变为 [16, 14, 14]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 阶段 2：第二层特征提取（第二层卷积）
        # 输入 16 通道，输出 32 通道 [32, 14, 14] -> 池化后变为 [32, 7, 7]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 阶段 3：分类头 (展平 + 全连接)
        # 32 个通道 * 7 * 7 高宽 = 1568
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  #第一层全连接，不能直接降到10个类别，否则特征损失严重
        self.relu3 = nn.ReLU()                            #再将维度之后再加一层激活函数，128的基础上非线性表达能力强一点，10个的话谈不上什么非线性表达能力了
        self.fc2 = nn.Linear(128, 10)  # 输出 10 个类别的预测得分，最终分类全连接

    def forward(self, x): #前向传播
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


model = FashionCNN().to(device)
print("\n网络结构定义成功：\n", model)

# 4. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器，学习率 0.001

# 5. 核心训练循环 (Training Loop)
num_epochs = 2  # 快速体验，先跑 2 轮
print("\n开始训练模型...")

for epoch in range(num_epochs):  #内部包含全流程而不只是cnn这个网络
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # --- 核心五步连招 ---
        # 1. 前向传播 用cnn
        outputs = model(images)
        # 2. 计算损失
        loss = criterion(outputs, labels)
        # 3. 梯度清零
        optimizer.zero_grad()
        # 4. 反向求导计算梯度
        loss.backward()
        # 5. 优化器更新参数
        optimizer.step()

        running_loss += loss.item()
        # 每隔 200 个 batch 打印一次平均 Loss
        if (i + 1) % 200 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 200:.4f}")
            running_loss = 0.0

print("训练完成！")

# 6. 在测试集上评估模型的预测准确率
correct = 0
total = 0
# 测试评估阶段不需要求导计算梯度，节省内存与计算
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # 取 10 个类别中得分最大的索引作为最终预测结果
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n模型在 10,000 张测试图片上的准确率为: {accuracy:.2f}%")

# 7. 保存训练好的参数权重
torch.save(model.state_dict(), "fashion_cnn.pth")
print("模型权重已成功保存为 fashion_cnn.pth！")