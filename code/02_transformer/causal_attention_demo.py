"""
8、
【模块名称】因果自注意力机制 (Causal / Masked Attention)
【解决痛点】自回归生成模型（如 GPT、LLM）在生成下一个词时，严禁偷看未来的词。
【核心流程】
  1. 计算原始打分矩阵 Q * K^T / sqrt(d_k) -> 形状 [B, N, N]
  2. 生成上三角掩码矩阵 (Upper Triangular Mask)
  3. 用 masked_fill 将未来时间步填充为 -inf (负无穷大)
  4. 经过 Softmax 后，未来的注意力权重精确变为 0% (形成严格下三角)
【张量流动】输入打分: [1, 4, 4] -> 掩码打分: [1, 4, 4] -> 权重矩阵: [1, 4, 4] (下三角)
"""

import torch
import math

# 模拟 4 个词，每个词特征维度为 8
seq_len = 4
embed_dim = 8

# 随机生成 Q 和 K
Q = torch.randn(1, seq_len, embed_dim)
K = torch.randn(1, seq_len, embed_dim)

# 1. 原始打分矩阵: [1, 4, 4]
scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(embed_dim)
print("1. 原始未遮挡的打分矩阵 (包含偷看未来的分数):\n", scores[0])

# 2. 构造因果掩码：对角线及以下为 False (保留)，严格上三角为 True (需要遮挡)
# torch.triu(..., diagonal=1) 会提取上三角部分
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print("\n2. 因果遮挡掩码 (True 代表未来需要屏蔽的位置):\n", mask)

# 3. 核心绝招：把未来位置全部填上负无穷大 (-inf)
masked_scores = scores.masked_fill(mask, float('-inf'))
print("\n3. 注入负无穷大后的打分矩阵:\n", masked_scores[0])

# 4. Softmax 归一化为注意力权重
attn_weights = torch.softmax(masked_scores, dim=-1)
print("\n4. 最终注意力权重矩阵 (严格的下三角概率分布):\n", attn_weights[0])