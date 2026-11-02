"""
1、看 3D 高斯的参数结构与空间投影
【模块名称】3D 高斯显式表征与参数构造 (3D Gaussian Representation Toy)
【解决痛点】理解 3DGS 中每个点元的数据结构：中心位置、缩放尺度、旋转四元数、不透明度与颜色。
【核心公式】
  1. 协方差矩阵构建: Sigma = R * S * S^T * R^T (确保半正定)
  2. 显卡可微光栅化前提: 每一个三维高斯都是由独立可学习张量拼接而成的实体
【张量流动】初始化 N 个高斯 (例如 N=100) -> 构造四元数旋转矩阵与尺度矩阵 -> 输出协方差形状 [N, 3, 3]
"""

import torch
import torch.nn as nn


class ToyGaussianModel(nn.Module):
    def __init__(self, num_points=100):
        super().__init__()
        self.num_points = num_points
        # 第22~39行的参数定义：
        # 看清每个高斯包含哪些具体物理属性。
        # 3DGS并没有庞大的全连接层或多层卷积，这些张量本身就是场景模型！
        # ==========================================================
        # 【重点解剖行 1】：3D 高斯核心参数张量定义
        # ==========================================================
        # 1. 空间三维中心位置 mu: [N, 3] (x, y, z)
        self._xyz = nn.Parameter(torch.randn(num_points, 3))

        # 2. 空间三维缩放尺度 Scale (对数空间存储以防负数): [N, 3] (sx, sy, sz)   S矩阵通过对数指数保持长宽高恒正
        self._scaling = nn.Parameter(torch.zeros(num_points, 3))

        # 3. 三维旋转朝向 (单位四元数 q = [w, x, y, z]): [N, 4]    R矩阵由单位四元数构造的纯旋转矩阵，，，这样构造出来的协方差矩阵才能保证为正定！
        self._rotation = nn.Parameter(torch.zeros(num_points, 4))
        self._rotation.data[:, 0] = 1.0  # 初始化为单位四元数 [1, 0, 0, 0] (无旋转)

        # 4. 不透明度 (激活前，通过 Sigmoid 映射到 0~1): [N, 1]
        self._opacity = nn.Parameter(torch.zeros(num_points, 1))

        # 5. 基础 RGB 颜色 (实际中用球谐函数系数): [N, 3]
        self._features_dc = nn.Parameter(torch.rand(num_points, 3))

    def get_rotation_matrix(self):
        """将四元数正则化后转为 3x3 旋转矩阵 R"""
        # 归一化四元数
        q = torch.nn.functional.normalize(self._rotation, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # 四元数转 3x3 旋转矩阵标准公式
        R = torch.zeros(self.num_points, 3, 3, device=q.device)
        R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
        return R

    # ==========================================================
    # 【重点解剖行 2】：计算三维协方差矩阵 Sigma = R * S * S^T * R^T
    # ==========================================================
    def get_covariance(self):
        # 尺度还原 (取 exp 保证半轴长恒正)
        s = torch.exp(self._scaling)  # [N, 3]
        S = torch.diag_embed(s)  # 对角阵 [N, 3, 3]

        # 获取旋转矩阵
        R = self.get_rotation_matrix()  # [N, 3, 3]

        # M = R * S
        M = torch.bmm(R, S)

        # Sigma = M * M^T = R * S * S^T * R^T
        covariance = torch.bmm(M, M.transpose(1, 2))
        return covariance


if __name__ == "__main__":
    num_gaussians = 5
    model = ToyGaussianModel(num_points=num_gaussians)

    print(f"--- 成功构建 {num_gaussians} 个三维高斯点元模型 ---")
    print("1. 空间中心位置坐标 _xyz 形状:", model._xyz.shape)
    print("2. 尺度特征 _scaling 形状:     ", model._scaling.shape)
    print("3. 姿态四元数 _rotation 形状:   ", model._rotation.shape)
    print("4. 不透明度 _opacity 形状:     ", model._opacity.shape)
    print("5. 颜色外观 _features_dc 形状: ", model._features_dc.shape)

    cov = model.get_covariance()
    print("\n【核心矩阵验证】构建出的 3D 协方差矩阵形状:", cov.shape)
    print("第 0 号高斯的 3x3 协方差矩阵对角线元素 (方差主轴长):\n", cov[0].diagonal())