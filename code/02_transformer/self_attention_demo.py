# 2、
# a:明确输入张量如何生成的QKV矩阵
# b:
import torch
import torch.nn as nn
import math

# 1. 模拟接上一节的输出：Batch=1, 序列长度=16 (图块数), 特征维度=64
batch_size = 1
seq_len = 16
embed_dim = 64

# 随机生成一个输入序列张量 x
x = torch.randn(batch_size, seq_len, embed_dim)
print(f"输入张量形状 [Batch, Seq_Len, Embed_Dim]: {x.shape}")

# 2. 定义把 x 变成 Q, K, V 的三个线性投影层
# 每一个图块都通过全连接变换，分别生成属于自己的 Q, K, V
w_q = nn.Linear(embed_dim, embed_dim)
w_k = nn.Linear(embed_dim, embed_dim)
w_v = nn.Linear(embed_dim, embed_dim)

Q = w_q(x)  # [1, 16, 64]
K = w_k(x)  # [1, 16, 64]
V = w_v(x)  # [1, 16, 64]

# 3. 核心计算步骤一：计算打分矩阵 Q * K^T
# K.transpose(1, 2) 把形状从 [1, 16, 64] 变成 [1, 64, 16]
# [1, 16, 64] 矩阵乘 [1, 64, 16] -> 得到 [1, 16, 16] 的注意力得分矩阵
scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(embed_dim)
print(f"\n1. 注意力打分矩阵形状 [Batch, Seq_Len, Seq_Len]: {scores.shape}")

# 4. 核心计算步骤二：Softmax 归一化为概率分布
attn_weights = torch.softmax(scores, dim=-1)
# 验证一下：每一行相加的总和是否为 1
print(f"2. 第一个图块对全场 16 个图块的关注权重之和: {attn_weights[0, 0].sum().item():.2f}")

# 5. 核心计算步骤三：加权聚合 Value (Attn_Weights * V)
# [1, 16, 16] 矩阵乘 [1, 16, 64] -> 输出 [1, 16, 64]
out = torch.matmul(attn_weights, V)
print(f"\n3. 经过自注意力聚合后的最终输出形状: {out.shape}")