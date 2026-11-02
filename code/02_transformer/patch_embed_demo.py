#1、看张量怎么变形。
#patch-embedding

import torch
import torch.nn as nn

# 1. 模拟输入一张单通道图片：Batch=1, 通道=1, 高=28, 宽=28 (Fashion-MNIST 大小)
x = torch.randn(1, 1, 28, 28)
print("原始图片输入形状 [Batch, C, H, W]:", x.shape)

# 2. 定义切片参数
patch_size = 7      # 每个切片长宽为 7x7
embed_dim = 64      # 把每个切片映射成 64 维的特征向量

# 3. 天才设计：在代码中，通常直接用一个“卷积核大小=patch_size, 步长=patch_size”的卷积来实现切片！
# 为什么？因为步长等于核大小时，卷积核刚好不重叠地一块一块跳着切！
# 这行卷积同时完成了“切片”和“映射”：
# kernel_size=7, stride=7 -> 负责“切片”（以 7 为步长一块一块跳着切）
# in_channels=1, out_channels=64 -> 负责“映射”（把 49 个像素映射为 64 维特征）
proj = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)   #怎么把 49 变成 64 的底层矩阵乘法

# 执行卷积切片
feat = proj(x)
print("\n经过 Patch 卷积后的形状 [Batch, embed_dim, H/7, W/7]:", feat.shape)

# 4. 展平并调整维度，变成 Transformer 喜欢的序列格式 [Batch, 序列长度, embed_dim]
# flatten(2) 把最后两个维度 (4, 4) 展平成 16
# transpose(1, 2) 把通道维度和序列维度换个位置
tokens = feat.flatten(2).transpose(1, 2)
print("最终送入 Transformer 的 Token 形状 [Batch, 序列长度(词数), 特征维度]:", tokens.shape)