"""
1、扩散加噪
【模块名称】Diffusion 正向加噪过程 (Forward Noising Process)
【核心目的】搞懂加噪如何通过闭式解公式一步到位，不需要循环 1000 次。
【核心公式】x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
【张量流动】干净图片 [1, 1, 28, 28] + 噪声 [1, 1, 28, 28] -> 任意时刻加噪图 [1, 1, 28, 28]
"""

import torch
import torch.nn as nn
import math


class LinearNoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        # 1. 定义每一步加噪的比例 beta (从很小逐渐变大)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        # 2. 保留原图的比例 alpha = 1 - beta
        self.alphas = 1.0 - self.betas
        # 3. 累乘系数 alpha_bar: 决定经过 t 步后，还残留多少原图信号
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    # ==========================================================
    # 【重点解剖行】：正向加噪的核心函数 (闭式解公式)
    # ==========================================================
    def add_noise(self, original_image, noise, t):
        """
        original_image: x_0 (原始图片)
        noise: epsilon (随机采样的高斯噪声)
        t: 时间步 (取值范围 0 到 999)
        """
        # 取出第 t 步对应的 alpha_bar 系数
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_cumprod[t])

        # 【核心观察点】：按比例将原图与噪声线性融合！
        noisy_image = sqrt_alpha_bar * original_image + sqrt_one_minus_alpha_bar * noise
        return noisy_image


if __name__ == "__main__":
    # 模拟一张 28x28 的干净单通道图像 (像素值在 [-1, 1] 之间)
    x_0 = torch.ones(1, 1, 28, 28) * 0.8  # 模拟一块干净的灰色布料
    noise = torch.randn_like(x_0)  # 采样同尺寸的标准高斯纯噪声

    scheduler = LinearNoiseScheduler(timesteps=1000)

    print("--- 观察不同时间步 t 下，原图信号与噪声信号的权重比例 ---")
    test_steps = [0, 100, 500, 999]
    for t in test_steps:
        # 计算第 t 步的融合图
        x_t = scheduler.add_noise(x_0, noise, t)

        # 提取当前时间步的两个权重系数
        signal_rate = torch.sqrt(scheduler.alphas_cumprod[t]).item()
        noise_rate = torch.sqrt(1.0 - scheduler.alphas_cumprod[t]).item()

        print(
            f"时间步 t = {t:3d} | 原图保留比例: {signal_rate:.4f} | 噪声掺入比例: {noise_rate:.4f} | 加噪图均值: {x_t.mean().item():.4f}")