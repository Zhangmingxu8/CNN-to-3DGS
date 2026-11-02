"""
4、CFG引导 
【模块名称】无分类器引导机制 (Classifier-Free Guidance, CFG Toy Demo)
【解决痛点】解决无条件生成不可控、以及纯条件生成对文本提示词贴合度不足的问题。
【核心公式】eps_final = eps_uncond + s * (eps_cond - eps_uncond)
【核心流程】
  1. 模拟一个能够接收文本条件向量 c 的条件 UNet
  2. 针对同一张加噪图 x_t，分别进行无条件推理 (c=None) 与条件推理 (c="红鞋")
  3. 计算两者的差值向量 (提示词带来的方向偏移)
  4. 乘以引导系数 s (如 s=7.5)，得到最终强引导的预测噪声
【张量流动】加噪图 [1, 1, 28, 28] + 条件向量 [1, 16] -> CFG 放大 -> 终验引导噪声 [1, 1, 28, 28]
"""

import torch
import torch.nn as nn


class ToyConditionalUNet(nn.Module):
    def __init__(self, cond_dim=16):
        super().__init__()
        # 图像特征主干
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        # 条件映射层：把 16 维的文本向量映射并注入到图像通道中
        self.cond_proj = nn.Linear(cond_dim, 16)
        self.out_conv = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, cond=None):
        h = self.relu(self.conv(x))

        # 如果提供了条件（有文本 Prompt），就把条件向量投影并加到特征图上
        if cond is not None:
            cond_bias = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)  # 变形为 [B, 16, 1, 1]
            h = h + cond_bias  # 核心：条件注入 (广播相加)

        return self.out_conv(h)


if __name__ == "__main__":
    model = ToyConditionalUNet(cond_dim=16)
    model.eval()

    # 模拟一张中间时刻的加噪图 x_t
    x_t = torch.randn(1, 1, 28, 28)

    # 1. 模拟文本编码器 (Text Encoder) 提炼出的提示词向量 (比如代表 "红色鞋子")
    text_condition = torch.randn(1, 16)

    with torch.no_grad():
        # ==========================================================
        # 【重点解剖行 1】：两次前向推理 (一次不带条件，一次带条件)
        # ==========================================================
        eps_uncond = model(x_t, cond=None)  # 无条件预测 (任由自由发散)
        eps_cond = model(x_t, cond=text_condition)  # 条件预测 (听从提示词指导)

        # 引导系数 CFG Scale
        cfg_scale = 7.5

        # ==========================================================
        # 【重点解剖行 2】：CFG 核心放大算子，对文字引导产生的偏移量放大，达到文字更有效果的结果
        # ==========================================================
        eps_final = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    print("--- CFG 引导结果对比 ---")
    print(f"无条件预测噪声方差: {eps_uncond.var().item():.4f}")
    print(f"带条件预测噪声方差: {eps_cond.var().item():.4f}")
    print(f"经过 CFG (s={cfg_scale}) 强引导放大后的噪声方差: {eps_final.var().item():.4f}")
    print(f"最终输出引导噪声形状保持一致: {eps_final.shape}")