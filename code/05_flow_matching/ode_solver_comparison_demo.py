"""
8、一阶欧拉法与中点法精度对比
【模块名称】常微分方程求解器精度对比 (Euler vs Midpoint / RK2)
【解决痛点】一阶欧拉法在曲线流场中容易因切线外推过大产生截断误差；二阶中点法通过中间采样点速度修正，使轨迹误差下降一个数量级。
【核心流程】
  1. 设定一个带曲率的速度场函数 v(x, t)
  2. 使用一阶欧拉法推进单步: x_new = x + v(x, t) * dt
  3. 使用二阶中点法推进单步:
     - v1 = v(x, t)
     - x_mid = x + v1 * (dt / 2)
     - v_mid = v(x_mid, t + dt / 2)
     - x_new = x + v_mid * dt
【张量流动】输入状态 [1, 1, 28, 28] -> 单步大步长积分 -> 误差分析
"""

import torch
import torch.nn as nn

# 1. 模拟一个带非线性曲率的真实流场: v = -2 * x * t + sin(x)
def true_velocity_field(x, t):
    return -2.0 * x * t + torch.sin(x)

# 2. 模拟高精度的真实真值基准 (用极其微小步长的欧拉法迭代 1000 步作为 Ground Truth)
def compute_ground_truth(x_start, t_start, t_end, sub_steps=1000):
    x = x_start.clone()
    dt = (t_end - t_start) / sub_steps
    t = t_start
    for _ in range(sub_steps):
        v = true_velocity_field(x, t)
        x = x + v * dt
        t = t + dt
    return x

if __name__ == "__main__":
    torch.manual_seed(42)
    x_init = torch.randn(1, 1, 28, 28)
    t_start = 1.0
    t_end = 0.5  # 大跨步反向采样: dt = -0.5
    dt = t_end - t_start

    # 计算真实轨迹基准
    x_true = compute_ground_truth(x_init, t_start, t_end)

    # ==========================================================
    # 【求解器 1】：一阶欧拉法 (仅计算 1 次速度，直接大步跨越)
    # ==========================================================
    v_euler = true_velocity_field(x_init, t_start)
    x_euler = x_init + v_euler * dt

    # ==========================================================
    # 【求解器 2】：二阶中点法 (Midpoint，两次测速)
    # ==========================================================
    # 步骤 A: 探测中点位置
    v1 = true_velocity_field(x_init, t_start)
    x_mid = x_init + v1 * (dt / 2.0)
    t_mid = t_start + dt / 2.0

    # 步骤 B: 提取中点速度并执行最终位移
    v_mid = true_velocity_field(x_mid, t_mid)
    x_midpoint = x_init + v_mid * dt

    # 误差计算 (MSE 越小说明越贴近真实平滑曲线)
    error_euler = torch.mean((x_euler - x_true) ** 2).item()
    error_midpoint = torch.mean((x_midpoint - x_true) ** 2).item()

    print(f"大步长 (dt = {dt}) 下的单步截断误差对比:")
    print(f"一阶欧拉法 (Euler) MSE 误差:   {error_euler:.6f}")
    print(f"二阶中点法 (Midpoint) MSE 误差: {error_midpoint:.6f}")
    print(f"中点法精度提升倍数:            {error_euler / error_midpoint:.2f} 倍")