"""
5、计算 0 阶与 1 阶球谐函数的视角变色
【模块名称】球谐函数 (Spherical Harmonics, SH) 视角变色极简模拟
【解决痛点】死板 RGB 无法表达高光与反光；通过视线方向 d 与 SH 基函数加权，使物体随相机观察角度动态改变颜色。
【核心公式】
  1. 视线方向单位向量: dir = (camera_pos - gaussian_pos) / norm
  2. 0 阶基函数 (常数底色): Y_0 = 0.282095
  3. 1 阶基函数 (三轴方向梯度): Y_1 = 0.488603 * y, 0.488603 * z, 0.488603 * x
  4. 最终颜色: Color = SH_0 * Y_0 + sum(SH_1_i * Y_1_i)
【张量流动】观察视角视线向量 [3] -> 投影至 SH 基函数 -> 输出不同视角下的最终 RGB [3]
"""

import torch
import math

# 球谐基函数的归一化常数 (来自拉普拉斯方程正交基推导)
C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
C1 = 0.4886025119029199  # sqrt(3) / (2 * sqrt(pi))


def compute_color_from_sh(view_dir, sh_0, sh_1):
    """
    根据给定的视线单位方向向量 view_dir 与 0/1 阶 SH 系数，计算当前观察到的颜色
    :param view_dir: [3] (dx, dy, dz) 单位视线向量
    :param sh_0: [3] 0阶参数 (RGB 基础固有色)
    :param sh_1: [3, 3] 1阶参数 (3个基函数，每个基函数有 RGB 3个通道)
    """
    # 1. 0 阶贡献 (无视角度，各向同性的底色)
    rgb = sh_0 * C0

    # 2. 1 阶贡献 (沿 x, y, z 轴的视角动态明暗)
    x, y, z = view_dir[0], view_dir[1], view_dir[2]

    # 1 阶的 3 个基函数值
    y_1_0 = C1 * y
    y_1_1 = C1 * z
    y_1_2 = C1 * x

    # 加权累加到 RGB 颜色上
    rgb = rgb + sh_1[0] * y_1_0 + sh_1[1] * y_1_1 + sh_1[2] * y_1_2

    # 加上 0.5 偏置并截断在 0~1 (保证颜色合法)
    rgb = torch.clamp(rgb + 0.5, 0.0, 1.0)
    return rgb


if __name__ == "__main__":
    # 模拟一个高斯：
    # 0 阶系数 (基础底色为深红色)
    sh_0 = torch.tensor([1.5, 0.0, 0.0])

    # 1 阶系数 (在 Z 轴方向设计了一个强烈的绿色反光)
    sh_1 = torch.tensor([
        [0.0, 0.0, 0.0],  # 对应 y 分量
        [0.0, 2.0, 0.0],  # 对应 z 分量 (当视线沿 Z 轴看时激发出强烈绿色)
        [0.0, 0.0, 0.0]  # 对应 x 分量
    ])

    # 模拟相机从两个截然不同的角度观察这同一个高斯：
    # 视角 A: 正对着看 (沿 Z 轴正方向，view_dir = [0, 0, 1])
    dir_A = torch.tensor([0.0, 0.0, 1.0])
    color_A = compute_color_from_sh(dir_A, sh_0, sh_1)

    # 视角 B: 侧着看 (沿 X 轴方向，view_dir = [1, 0, 0])
    dir_B = torch.tensor([1.0, 0.0, 0.0])
    color_B = compute_color_from_sh(dir_B, sh_0, sh_1)

    # 视角 C: 背着看 (沿 Z 轴负方向，view_dir = [0, 0, -1])
    dir_C = torch.tensor([0.0, 0.0, -1.0])
    color_C = compute_color_from_sh(dir_C, sh_0, sh_1)

    print("--- 同一个 3D 高斯在不同观察视角下的色彩变化 ---")
    print(f"视角 A (正对 Z 轴看)   -> RGB 颜色: {color_A.numpy().round(3)} (激发出亮黄色/绿色高光)")
    print(f"视角 B (从侧面 X 轴看) -> RGB 颜色: {color_B.numpy().round(3)} (还原为纯红底色)")
    print(f"视角 C (从背后 -Z 轴看) -> RGB 颜色: {color_C.numpy().round(3)} (背光暗色)")