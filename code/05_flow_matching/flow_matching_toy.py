"""
6、
【模块名称】流匹配基础原理 (Flow Matching Toy Demo)
【解决痛点】传统扩散模型曲线采样轨迹长、推理慢；Flow Matching 构建直线常微分方程 (ODE)，推理快、数学形式极简。
【核心公式】
  1. 路径插值: x_t = (1 - t) * x_0 + t * x_1
  2. 真实目标速度: v_target = dx_t / dt = x_1 - x_0
  3. 训练目标: Loss = MSE(v_pred, v_target)
【张量流动】原图 x_0 [1, 1, 28, 28] + 噪声 x_1 [1, 1, 28, 28] -> 线性插值与恒定速度场构建
"""

import torch
import torch.nn as nn
import torch.optim as optim


# 1. 极简速度预测网络 (模拟 DiT / Vector Field Predictor)
class ToyFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x, t):
        # 实际模型中会将时间 t 编码注入，这里返回预测的速度向量场
        return self.net(x)


if __name__ == "__main__":
    # 模拟输入：真实干净图像 x_0，与纯高斯噪声 x_1
    x_0 = torch.randn(1, 1, 28, 28)  # 真实数据 (t=0)
    x_1 = torch.randn(1, 1, 28, 28)  # 纯噪声数据 (t=1)

    # 随机采样一个时间点 t (范围在 0 到 1 之间)
    t = torch.tensor([0.4])

    # ==========================================================
    # 【重点解剖行 1】：直线插值路径 (Straight Path)，没有复杂的加噪公式
    # ==========================================================
    x_t = (1.0 - t) * x_0 + t * x_1

    # ==========================================================
    # 【重点解剖行 2】：解析真实速度 (Ground Truth Velocity)
    # ==========================================================
    v_target = x_1 - x_0

    # 模型前向与训练
    model = ToyFlowNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("--- 开始训练 Flow Matching 速度场网络 ---")
    for step in range(5):
        optimizer.zero_grad()

        # 让网络根据当前位置 x_t 和时间 t 预测前进速度
        v_pred = model(x_t, t)

        # 目标就是让网络预测的速度逼近两点之间的恒定直线速度！
        loss = criterion(v_pred, v_target)
        loss.backward()
        optimizer.step()

        print(f"训练步骤 [{step + 1}/5] | 速度场拟合 MSE Loss: {loss.item():.4f}")

    print("\n【轨迹验证】:")
    print(f"t=0 时的位置接近原图: {((1.0 - 0.0) * x_0 + 0.0 * x_1).mean().item():.4f}")
    print(f"t=1 时的位置完全是噪声: {((1.0 - 1.0) * x_0 + 1.0 * x_1).mean().item():.4f}")
    print("速度场输出形状与图像一致:", v_pred.shape)