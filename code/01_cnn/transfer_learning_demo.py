
#4、调用官方 ResNet-18（内部训练好的已经包含上千种物体，百万张图片） 做微调
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# 1. 设置计算设备
device = torch.device("cpu")
print(f"当前计算设备: {device}")

# 2. 准备数据预处理
# ResNet 官方模型要求输入是 3 通道 (RGB)，且推荐分辨率至少是 224x224
# 但为了让你在本地 CPU 上飞速跑通，我们这里只放大到 32x32，并把单通道灰度图复制成 3 通道
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),  # 灰度图转为 3 通道
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("加载 Fashion-MNIST 数据集...")
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# 仅取前 2000 张图片做快速微调体验
train_subset = torch.utils.data.Subset(train_dataset, range(2000))
test_subset = torch.utils.data.Subset(test_dataset, range(500))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

# 3. 核心步骤：加载官方预训练好的 ResNet-18
print("\n正在加载官方预训练 ResNet-18 权重...")
# 第一次运行会自动下载约 44MB 的官方预训练模型权重
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 4. 核心步骤：冻结所有卷积主干层参数（不参与梯度更新）
for param in model.parameters():
    param.requires_grad = False

# 5. 核心步骤：手术式替换最后一层分类头 (model.fc)
# 查看原先的输入特征维度 (in_features 是 512)
in_features = model.fc.in_features
# 换成我们自己的全连接层，输出类别改为 10 (Fashion-MNIST 的 10 种衣服)
model.fc = nn.Linear(in_features, 10)
# 新换上去的这层 fc，默认 requires_grad = True，只有它会参与训练！

model = model.to(device)
print(f"已成功替换分类头: 接收 {in_features} 维特征，输出 10 类别")

# 6. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
# 关键细节：优化器只传入 model.fc 的参数！
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

# 7. 训练只更新分类头 (1 个 Epoch 即可)
print("\n开始快速微调训练 (仅训练最后一层)...")
model.train()
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        print(f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# 8. 在测试集上评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n微调后在测试集上的准确率: {100 * correct / total:.2f}%")
