#2、cnn网络搭建后 调用模型 对数据库中的图片进行识别

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 解决画图时中文字体可能乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 1. 重新声明一样的网络架构（保证结构一致）
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 2. 实例化模型  并加载刚保存好的权重文件
model = FashionCNN()
model.load_state_dict(torch.load("fashion_cnn.pth", map_location=torch.device('cpu')))
model.eval()  # 切换为评估/推理模式（不更新梯度）
print("成功加载已训练好的权重文件: fashion_cnn.pth")

# 3. 准备测试集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform)

classes = ['T恤', '裤子', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']

# 4. 从测试集里随便抽取一张图片做推理
sample_idx = 999  # 可以随意改这个索引换不同的图片测试
image_tensor, true_label = test_dataset[sample_idx]

# 增加一个 Batch 维度: 从 [1, 28, 28] 变成 [1, 1, 28, 28]
input_tensor = image_tensor.unsqueeze(0)

# 5. 推理计算
with torch.no_grad():
    output = model(input_tensor)
    # 取得分最大的类别索引
    # print("10个类别的原始打分(Logits):\n", output)
    # print("经过Softmax后的概率分布:\n", torch.softmax(output, dim=1))
    _, predicted_idx = torch.max(output, 1)

pred_name = classes[predicted_idx.item()]
true_name = classes[true_label]

print(f"\n样本索引 {sample_idx}:")
print(f"真实标签: {true_name}")
print(f"模型预测: {pred_name}")

# 6. 把这张图片弹窗画出来
# 还原归一化以便正常显示灰度像素
display_image = image_tensor.squeeze().numpy() * 0.5 + 0.5
plt.figure(figsize=(3, 3))
plt.imshow(display_image, cmap='gray')
plt.title(f"真实: {true_name} | 预测: {pred_name}")
plt.axis('off')
plt.show()