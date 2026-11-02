"""
3、
【模块名称】3D 高斯投影与 2D Alpha Blending 极简模拟 (EWA Splatting Toy)
【解决痛点】亲手验证 3D 协方差如何通过视角变换与投影雅可比矩阵映射为 2D 屏幕协方差，并进行前向像素着色。
【核心公式】
  1. 屏幕协方差投影: Sigma_2D = J * W * Sigma_3D * W^T * J^T
  2. 2D 高斯强度采样: G(x) = exp(-0.5 * (x - mu)^T * Sigma_2D^{-1} * (x - mu))
  3. Alpha Blending 颜色合成: C = sum(c_i * alpha_i * T_i)
【张量流动】3D 协方差 [3, 3] -> 相机投影 -> 2D 协方差 [2, 2] -> 渲染一个 32x32 屏幕画面
"""

import torch
import torch.nn as nn
import math


def project_3d_to_2d(mean_3d, cov_3d, W, focal_length=100.0):
    """
    极简相机投影函数：将 3D 位置与 3D 协方差投影到 2D 屏幕
    """
    # 1. 转换到相机坐标系: p_cam = W * p_world
    # 为简化，假定 W 为单位阵 (相机位于原点往前看)
    p_cam = mean_3d
    z = p_cam[2]  # 深度距离

    # 2. 计算透视投影在当前点的雅可比矩阵 J (对 x/z, y/z 求导)
    # 负责近大远小投影
    J = torch.tensor([
        [focal_length / z, 0.0, -(focal_length * p_cam[0]) / (z ** 2)],
        [0.0, focal_length / z, -(focal_length * p_cam[1]) / (z ** 2)]
    ])  # 形状: [2, 3]

    # ==========================================================
    # 【重点解剖行 1】：EWA 投影公式 Sigma_2D = J * W * Sigma_3D * W^T * J^T
    # ==========================================================
    # 假定旋转变换已被整合，这里计算 J * cov_3d * J^T
    cov_2d = torch.mm(J, torch.mm(cov_3d, J.t()))

    # 加上 0.3 的低通滤波 (防止椭圆在屏幕上缩成无限小的单点造成走样闪烁)
    cov_2d[0, 0] += 0.3
    cov_2d[1, 1] += 0.3

    # 屏幕投影中心坐标 (像素坐标)
    screen_mean = torch.tensor([
        focal_length * p_cam[0] / z + 16.0,  # 居中偏移到 32x32 画布中心
        focal_length * p_cam[1] / z + 16.0
    ])

    return screen_mean, cov_2d


if __name__ == "__main__":
    # 1. 定义一个空间中的 3D 椭球高斯
    mean_3d = torch.tensor([0.0, 0.0, 5.0])  # 位于正前方 z=5 处
    # 定义一个稍微拉长且带旋转的 3x3 协方差矩阵
    cov_3d = torch.tensor([
        [1.0, 0.2, 0.0],
        [0.2, 0.5, 0.0],
        [0.0, 0.0, 1.0]
    ])
    gaussian_color = torch.tensor([1.0, 0.0, 0.0])  # 鲜艳红色
    opacity = 0.8  # 不透明度 80%

    # 2. 执行 EWA 投影，得到 2D 椭圆属性
    W = torch.eye(3)
    screen_mean, cov_2d = project_3d_to_2d(mean_3d, cov_3d, W)
    print("--- 3D 高斯投影至 2D 屏幕成功 ---")
    print("2D 屏幕中心坐标 (像素):", screen_mean.numpy())
    print("2D 屏幕协方差矩阵 (2x2):\n", cov_2d.numpy())

    # ==========================================================
    # 【重点解剖行 2】：光栅化像素着色 (渲染一个 32x32 分辨率的小画幅)
    # ==========================================================
    H, W_canvas = 32, 32
    rendered_image = torch.zeros(H, W_canvas, 3)
    cov_2d_inv = torch.inverse(cov_2d)

    # 遍历屏幕像素 (实际 CUDA 内核中是几万个线程并行运行)
    for y in range(H):
        for x in range(W_canvas):
            pixel_coord = torch.tensor([float(x), float(y)])
            diff = pixel_coord - screen_mean

            # 计算 2D 马氏距离: d^2 = (p - mu)^T * Sigma^{-1} * (p - mu)
            power = -0.5 * torch.dot(diff, torch.mv(cov_2d_inv, diff))

            if power < 0:  # 在椭圆衰减范围内
                weight = math.exp(power) * opacity
                rendered_image[y, x] = weight * gaussian_color

    print(f"\n画面光栅化渲染完毕！渲染张量形状: {rendered_image.shape}")
    print(f"中心像素 (16, 16) 处的 RGB 颜色值: {rendered_image[16, 16].numpy()}")


# 五、 本次代码的【两个核心学习位置】
# 第 33 行：cov_2d = torch.mm(J, torch.mm(cov_3d, J.t()))：原本是 [3, 3] 的空间矩阵，经过雅可比矩阵 $J$（[2, 3]）左右夹击相乘，降维变成了 [2, 2] 的屏幕矩阵。这就是 3D 到 2D 拍扁的物理实现。
# 第 73 行：马氏距离衰减 power = -0.5 * diff^T * cov_2d_inv * diff：越靠近椭圆中心，衰减越小、颜色越浓；离中心越远，指数衰减为 0。