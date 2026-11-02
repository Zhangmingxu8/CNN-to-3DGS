#3、残差块问题

import torch
import torch.nn as nn


# 1. 定义一个最经典的残差块 (Residual Block)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 卷积主干分支 F(x)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # 快捷直通分支 (Shortcut / Skip Connection)
        self.shortcut = nn.Sequential()
        # 如果步长不为 1 (长宽变了)，或者输入输出通道不相等，输入 x 必须经过 1x1 卷积调整形状
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        identity = x  # 备份输入
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # 核心灵魂：主干输出 F(x) + 直通输入 x
        out = out + self.shortcut(identity)  #直接让卷积层可以达到1k层的理论
        out = self.relu(out)
        return out


# 2. 模拟测试：测试通道数相同且步长为 1 的情况（最纯粹的恒等映射）
block_plain = ResidualBlock(in_channels=16, out_channels=16, stride=1)
x1 = torch.randn(1, 16, 28, 28)
out1 = block_plain(x1)
print("测试 1 - 恒等残差块输出形状:", out1.shape)  #恒等残差块输出形状: torch.Size([1, 16, 28, 28]) x


# 3. 模拟测试：测试降采样且通道翻倍的情况 (16通道 -> 32通道, 步长 2 尺寸砍半)
#步长为2，空间分辨率减半，通过增加通道数，弥补为了扩大感受野的同时信息丢失的情况
#经典骨干网络都遵循这个规律：在长宽减半的时候，通道数翻倍，保证特征容量丰富
block_downsample = ResidualBlock(in_channels=16, out_channels=32, stride=2)
x2 = torch.randn(1, 16, 28, 28)
out2 = block_downsample(x2)
print("测试 2 - 降采样残差块输出形状:", out2.shape)   #降采样残差块输出形状: torch.Size([1, 32, 14, 14]) F(x)
