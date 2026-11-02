#6、MHA
"""
【模块名称】Multi-Head Attention (多头自注意力机制)
【解决痛点】单头注意力只能关注一种关系；多头机制让模型同时从多个角度（颜色/轮廓/语义）并行观察全图，且总参数量/计算量不增加。
【核心流程】
  1. 线性映射得到 Q, K, V
  2. 维度拆解 (Split Heads): 将 64 维拆为 4 个头，每个头 16 维 -> [B, 4, N, 16]
  3. 各头独立并行计算注意力打分 (Q * K^T / sqrt(d_k)) 与加权求和
  4. 维度拼接 (Concat Heads): 将 4 个头的 16 维拼回 64 维 -> [B, N, 64]
  5. 经过 Out Linear 线性投影输出
【张量流动】输入: [B, 17, 64] -> 注意力图: [B, 4, 17, 17] -> 最终输出: [B, 17, 64]
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度: 64 // 4 = 16

        # Q, K, V 投影矩阵与最终输出汇聚矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影: [Batch, Seq_Len, 64]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 核心操作：维度变形，拆成 4 个头！
        # [Batch, Seq_Len, 64] -> [Batch, Seq_Len, 4, 16] -> 转置为 [Batch, 4, Seq_Len, 16]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 4 个头同时并行算注意力打分
        # Q: [B, 4, N, 16], K^T: [B, 4, 16, N] -> scores: [B, 4, N, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权聚合 Value: [B, 4, N, N] @ [B, 4, N, 16] -> [B, 4, N, 16]
        context = torch.matmul(attn_weights, V)

        # 4. 把 4 个头的输出重新拼回 64 维
        # [B, 4, N, 16] -> [B, N, 4, 16] -> [B, N, 64]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 5. 最后过一层线性投影输出
        out = self.out_proj(context)
        return out, attn_weights


if __name__ == "__main__":
    # 模拟输入: 1 张图，17 个 Token (包含 CLS)，每个 Token 64 维
    x = torch.randn(1, 17, 64)
    mha = MultiHeadAttention(embed_dim=64, num_heads=4)
    output, weights = mha(x)

    print("Multi-Head Attention 运行成功！")
    print("输出特征形状 (维度保持不变):", output.shape)
    print("注意力权重形状 [Batch, Num_Heads, Seq_Len, Seq_Len]:", weights.shape)