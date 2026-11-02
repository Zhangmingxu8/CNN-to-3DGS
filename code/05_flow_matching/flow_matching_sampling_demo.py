"""
7、Flow Matching 欧拉法 10 步采样代码
【模块名称】Flow Matching 欧拉法 (Euler) 直线推理采样循环
【解决痛点】对比 Diffusion 繁琐的随机抖动采样；Flow Matching 仅通过初中物理公式 x = x + v * dt 直线推进，极简且高效。
【核心公式】
  1. 初始起点: x_1 ~ N(0, I) (纯噪声，对应时间 t=1.0)
  2. 时间步长: dt = 1.0 / steps (例如 10 步，dt = 0.1)
  3. 欧拉推进单步: x_{t - dt} = x_t + v_pred * (-dt)   (由于从 t=1 倒退向 t=0，位移为 -v * dt)
【张量流动】起点纯噪声 [1, 1, 28, 28] -> 10 步极简欧拉推进 -> 终点干净图像 [1, 1, 28, 28]
"""

import torch
import torch.nn as nn


# 1. 模拟一个训练好、能够预测瞬时速度场 v 的网络 (如 Flux / SD3 中的 DiT)
class PretrainedFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x, t):
        # 实际模型中会把连续的时间 t 融入特征中
        return self.conv(x)


# 2. 初始化环境
model = PretrainedFlowNet()
model.eval()

# 3. 核心起点：从纯高斯白噪声出发 (此时时间 t = 1.0)
x = torch.randn(1, 1, 28, 28)
print("0. 初始状态: 一张纯高斯随机噪声图 [1, 1, 28, 28] (对应 t=1.0)")

# 4. 设置欧拉采样的总步数 (仅需 10 步！)
steps = 10
# 生成从 1.0 倒退到 0.0 的时间网格: [1.0, 0.9, 0.8, ..., 0.1, 0.0]
timesteps = torch.linspace(1.0, 0.0, steps + 1)

print("\n--- 开始执行 Flow Matching 欧拉法直线采样 ---")
with torch.no_grad():
    for i in range(steps):
        t_current = timesteps[i]
        t_next = timesteps[i + 1]

        # 计算这一步的时间跨度 dt (由 1.0 退向 0.0，因此 dt 是负数，例如 -0.1)
        dt = t_next - t_current  # 例如 0.9 - 1.0 = -0.1

        # 步骤 A: 让神经网络预测当前位置的直线流动速度 v
        v_pred = model(x, t_current)

        # ==========================================================
        # 【重点解剖行】：最纯粹的欧拉积分单步推进 (初中物理: 位移 = 速度 * 时间)
        # ==========================================================
        x = x + v_pred * dt

        print(f"步骤 [{i + 1:2d}/{steps}] 从 t={t_current:.1f} 推进到 t={t_next:.1f} | "
              f"图像张量均值: {x.mean().item():.4f} | 方差: {x.var().item():.4f}")

print("\n--- 生成完成！---")
print("最终生成的图像张量形状:", x.shape)