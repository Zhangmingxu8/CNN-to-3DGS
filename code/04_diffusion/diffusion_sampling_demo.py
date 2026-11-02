"""
3、扩散推理
【模块名称】Diffusion 反向采样生成循环 (Denoising Sampling Loop)
【解决痛点】亲眼看懂扩散模型在推理时，是如何从纯随机高斯噪声逐步倒退还原出清晰图像的。
【核心流程】
  1. 采样一张纯高斯白噪声图片 x_T ~ N(0, I) 作为起点
  2. 启动倒序时间循环: for t in reversed(range(timesteps)):
  3. 用 UNet 预测当前时间步 t 的噪声 pred_noise
  4. 按照 DDPM 采样公式，扣除噪声并退回一步得到 x_{t-1}
  5. 到达 t=0，输出最终生成的干净图像
【张量流动】起点纯噪声 [1, 1, 28, 28] -> 逐步扣除噪声 (10步模拟) -> 终点图像 [1, 1, 28, 28]
"""

import torch
import torch.nn as nn
import math

# 1. 模拟一个已经训练好的极简 UNet (假定它能猜出噪声的主干)
class PretrainedToyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 用简单的卷积模拟特征提取与噪声估计
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x, t):
        # 实际模型中会把 t 编码注入，这里模拟网络输出预测噪声
        return self.conv(x)

# 2. 采样调度器 (为了演示直观，我们设总步长为 10 步，真实一般为 50~1000 步)
total_steps = 10
betas = torch.linspace(0.0001, 0.02, total_steps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

model = PretrainedToyUNet()
model.eval()

# 3. 核心起点：从纯正态分布采样一团【绝对的雪花屏】！
x = torch.randn(1, 1, 28, 28)
print("0. 初始状态: 一张纯粹的高斯随机噪声图 [1, 1, 28, 28]")

# 4. 反向去噪主循环 (从第 9 步倒着退回第 0 步)，反向去噪过程 训练神经网络，而不是正向加噪过程
print("\n--- 开始执行反向逐步去噪生成 ---")
with torch.no_grad():
    for t in reversed(range(total_steps)):
        # 步骤 A: 提取当前时间步的系数
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]

        # 步骤 B: 让神经网络猜测当前的噪声
        pred_noise = model(x, t)

        # ==========================================================
        # 【重点解剖行 1】：DDPM 核心减噪算子 (扣掉猜出的噪声)
        # ==========================================================
        coeff = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - coeff * pred_noise)  #x：加噪图；coeff * pred_noise：网络猜出来的噪声

        # 步骤 C: 如果还没到最后一步，加上微量随机扰动；到了第 0 步则不加
        if t > 0:
            noise_z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * noise_z
        else:
            x = mean

        print(f"时间步 t = {t:2d} 完成去噪 | 当前图像张量均值: {x.mean().item():.4f} | 方差: {x.var().item():.4f}")

print("\n--- 生成完成！---")
print("最终生成的图像张量形状:", x.shape)