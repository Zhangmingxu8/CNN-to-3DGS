# 06 · 3D Gaussian Splatting 实践

对应笔记：[`docs/06-3dgs.md`](../../docs/06-3dgs.md)

这一组 Demo 是"从零手写一条极简 3DGS 管线"，不依赖官方 CUDA 实现，纯 PyTorch 在 CPU 上跑。

## 文件（按管线顺序）

| 文件 | 对应环节 | 关键点 |
| :--- | :--- | :--- |
| `gaussian_representation_demo.py` | 参数与协方差 | $\Sigma = R S S^\top R^\top$ |
| `train_single_gaussian_toy.py` | 可微拟合 | 没有神经网络，参数本身就是模型 |
| `gaussian_splatting_2d_toy.py` | EWA 投影 + 着色 | $\Sigma_{2D}=J\Sigma J^\top$ + 马氏距离 |
| `spherical_harmonics_toy.py` | 视角相关颜色 | 0 阶常数 + 1 阶方向梯度 |
| `tile_sort_rasterization_toy.py` | 瓦片与排序 | 64 位键 `[TileID \| Depth]` |
| `adaptive_density_control_toy.py` | 增删策略 | Clone / Split / Prune |

## 运行

```bash
python code/06_3dgs/gaussian_representation_demo.py
python code/06_3dgs/train_single_gaussian_toy.py
python code/06_3dgs/gaussian_splatting_2d_toy.py
python code/06_3dgs/spherical_harmonics_toy.py
python code/06_3dgs/tile_sort_rasterization_toy.py
python code/06_3dgs/adaptive_density_control_toy.py
```

## 预期输出

```text
--- 成功构建 5 个三维高斯点元模型 ---
1. 空间中心位置坐标 _xyz 形状: torch.Size([5, 3])
...
【核心矩阵验证】构建出的 3D 协方差矩阵形状: torch.Size([5, 3, 3])
```

```text
迭代 [100/100] | L1 Loss: 0.000xx | 当前坐标: [8. 8.] | 当前颜色RGB: [1. 0. 0.]
```

## 说明

这些是**教学用玩具实现**：
- 单高斯拟合用的是 2D 屏幕坐标，不涉及相机投影；
- 2D 泼溅用双重 for 循环逐像素渲染，只为看清数学，真实管线是 CUDA 并行；
- 球谐只实现到 1 阶（真实 3DGS 用到 3 阶 / 48 个系数）。
