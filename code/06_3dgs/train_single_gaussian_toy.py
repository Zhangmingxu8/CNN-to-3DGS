"""
2、
【模块名称】单高斯参数可微拟合训练 (Train Single Gaussian Toy)
【解决痛点】亲眼验证 3DGS 如何在没有深度神经网络的情况下，通过图像损失直接对高斯三维几何属性 (坐标/颜色) 求导并迭代优化。
【核心流程】
  1. 设定一张目标真实图像 target_image (模拟相机拍到的真实红点)
  2. 初始化一个位置偏移、颜色发灰的 3D 高斯作为可学习参数
  3. 执行 2D 投影与可微高斯渲染
  4. 计算渲染图像与目标图像的 L1 Loss
  5. 执行 loss.backward()，直接更新高斯的位置与颜色张量
【张量流动】可学习参数 (mu, color) -> 屏幕光栅化 [16, 16, 3] -> 与真值算 Loss -> 梯度回传直接修正 mu 与 color
"""

import torch
import torch.nn as nn
import torch.optim as optim


class LearnableGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始中心位置：故意初始化在左上角 (4.0, 4.0)，目标在中心 (8.0, 8.0)
        self.screen_pos = nn.Parameter(torch.tensor([4.0, 4.0], dtype=torch.float32))  #没有权重矩阵，高斯的物理属性自身就是参数

        # 初始颜色：故意初始化为暗灰色 [0.2, 0.2, 0.2]，目标是纯红 [1.0, 0.0, 0.0]
        self.color = nn.Parameter(torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32))

    def render(self, H=16, W=16):
        # 构造网格坐标 [H, W, 2]
        y_coords = torch.arange(H, dtype=torch.float32)
        x_coords = torch.arange(W, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        pixel_coords = torch.stack([grid_x, grid_y], dim=-1)  # [16, 16, 2]

        # 计算每个像素到当前高斯中心的欧氏距离平方
        diff = pixel_coords - self.screen_pos
        dist_sq = torch.sum(diff ** 2, dim=-1)  # [16, 16]

        # 高斯平滑衰减 (方差 sigma^2 = 2.0)
        weight = torch.exp(-0.5 * dist_sq / 2.0).unsqueeze(-1)  # [16, 16, 1]

        # 渲染出当前画幅图像
        rendered = weight * self.color  # [16, 16, 3]
        return rendered


if __name__ == "__main__":
    torch.manual_seed(42)
    H, W = 16, 16

    # 1. 制造一张真实世界拍到的“目标真实照片” (中心在 (8.0, 8.0) 的亮红色光斑)
    y_c, x_c = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    target_dist = (x_c - 8.0) ** 2 + (y_c - 8.0) ** 2
    target_weight = torch.exp(-0.5 * target_dist / 2.0).unsqueeze(-1)
    target_image = target_weight * torch.tensor([1.0, 0.0, 0.0])  # 纯红目标

    # 2. 实例化可学习的高斯模型与优化器
    gaussian = LearnableGaussian()
    optimizer = optim.Adam(gaussian.parameters(), lr=0.1)
    criterion = nn.L1Loss()

    print("--- 训练前初始状态 ---")
    print(f"初始坐标: {gaussian.screen_pos.data.numpy()} (偏离目标 [8., 8.])")
    print(f"初始颜色: {gaussian.color.data.numpy()} (偏离纯红 [1., 0., 0.])")

    print("\n--- 开始执行梯度反向传播训练 (100 步) ---")
    for step in range(1, 101):
        optimizer.zero_grad()

        # 前向渲染
        pred_img = gaussian.render(H, W)

        # 计算渲染画面与真实照片的误差
        loss = criterion(pred_img, target_image)

        # ==========================================================
        # 【核心解剖行】：梯度直接回传给高斯的坐标和颜色！
        # ==========================================================
        loss.backward()   #损失函数只比较最终像素
        optimizer.step()

        if step % 20 == 0:
            print(f"迭代 [{step:3d}/100] | L1 Loss: {loss.item():.5f} | "
                  f"当前坐标: {gaussian.screen_pos.data.numpy().round(2)} | "
                  f"当前颜色RGB: {gaussian.color.data.numpy().round(2)}")

    print("\n--- 训练收敛完成 ---")
    print("最终学习到的坐标 (完美逼近 [8.0, 8.0]):", gaussian.screen_pos.data.numpy().round(2))
    print("最终学习到的颜色 (完美逼近纯红 [1.0, 0.0, 0.0]):", gaussian.color.data.numpy().round(2))