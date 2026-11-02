"""
7、
【模块名称】LayerNorm vs BatchNorm 维度计算对比
【核心目的】搞懂 BN 为什么跨 Batch 统计，而 LN 是在每个 Token 内部独立统计，理解 Transformer 为什么选用 LayerNorm。
【张量流动】
  输入: [Batch=2, Seq_Len=3, Embed_Dim=4]
  LN: 在最后一个维度 (dim=-1) 跨 4 个特征维度求均值和方差
  BN: 在 Batch 维度 (跨样本) 进行统计
"""

import torch
import torch.nn as nn

# 模拟 2 个句子/图片切片，每个序列长为 3，特征维度为 4
# 形状: [Batch=2, Seq_Len=3, Embed_Dim=4]
x = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0],
     [2.0, 4.0, 6.0, 8.0],
     [1.0, 1.0, 1.0, 1.0]],

    [[10.0, 20.0, 30.0, 40.0],
     [5.0, 5.0, 5.0, 5.0],
     [0.0, 0.0, 0.0, 0.0]]
])

print("原始输入形状 [Batch, Seq_Len, Embed_Dim]:", x.shape)

# 1. LayerNorm: 针对最后一个特征维度 embed_dim=4 进行自归一化
ln = nn.LayerNorm(normalized_shape=4)
out_ln = ln(x)

# 打印第一个样本的第一个 Token 归一化后的均值和方差
token_0 = out_ln[0, 0]
print("\n--- LayerNorm 结果 ---")
print(f"第一个 Token 归一化后的值: {token_0.detach().numpy()}")
print(f"Token 内部均值 (接近0): {token_0.mean().item():.4f}")
print(f"Token 内部方差 (接近1): {token_0.var(unbiased=False).item():.4f}")

# 2. BatchNorm1d: 需要把通道维度调到第 1 维 [Batch, Embed_Dim, Seq_Len]
bn = nn.BatchNorm1d(num_features=4)
# 转置以适配 BN: [2, 4, 3]
x_for_bn = x.transpose(1, 2)
out_bn = bn(x_for_bn).transpose(1, 2)

print("\n--- BatchNorm 结果 ---")
print("BN 必须跨越两个 Batch 样本一起统计，如果 Batch=1，BN 在训练时会直接报错！")