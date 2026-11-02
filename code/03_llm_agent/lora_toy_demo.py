"""
3、
【模块名称】LoRA (Low-Rank Adaptation) 底层数学机制直观演示
【解决痛点】全量微调大模型显存开销极大；LoRA 冻结主干权重，仅训练低维分解矩阵 A 和 B，极大节省显存。
【核心流程】
  1. 模拟一个被冻结的预训练大矩阵 W (例如 4096 维)
  2. 初始化低秩旁路: 矩阵 A (降维至 rank=8) 和 矩阵 B (升维至 4096)
  3. 前向传播: Output = x @ W + (x @ A @ B) * scaling
  4. 训练完成后可将 (B @ A) 直接折叠加回原矩阵，推理完全零额外延迟
【张量流动】输入: [1, 4096] -> 主路 + 旁路 -> 输出: [1, 4096]
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, lora_alpha=16):
        super().__init__()
        # 1. 原始预训练权重 (主干)：冻结，不参与训练！
        self.pretrained_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.pretrained_weight.requires_grad = False  # 关键：彻底冻结！

        # 2. LoRA 低秩旁路矩阵 A 和 B
        # A 负责从 4096 降维到 8 (高斯随机初始化)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        # B 负责从 8 升维回 4096 (全 0 初始化，保证刚开始旁路输出为 0，不破坏原模型性能)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.scaling = lora_alpha / rank

    def forward(self, x):
        # 主干路径 (不求导): x @ W^T
        base_out = nn.functional.linear(x, self.pretrained_weight)

        # LoRA 旁路 (参与求导微调): (x @ A^T @ B^T) * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        # 两路直接相加
        return base_out + lora_out


# 测试验证
in_dim = 4096
out_dim = 4096
rank = 8

layer = LoRALinear(in_dim, out_dim, rank=rank)

# 统计可训练参数量
trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in layer.parameters() if not p.requires_grad)

print(f"被冻结的原厂主干参数量: {frozen_params:,}")
print(f"实际需要训练的 LoRA 参数量: {trainable_params:,}")
print(f"可训练参数比例仅为: {trainable_params / (frozen_params + trainable_params) * 100:.2f}%")

# 模拟前向传播
dummy_input = torch.randn(1, in_dim)
output = layer(dummy_input)
print(f"\n输出张量形状保持完全一致: {output.shape}")