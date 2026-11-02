"""
4、模拟“克隆、分裂与修剪”决策
【模块名称】3DGS 自适应密度控制决策机制 (Adaptive Density Control Toy)
【解决痛点】直观理解算法如何根据梯度与尺度判定 Clone、Split 与 Pruning，实现高斯数量的动态增删。
【核心规则】
  1. 梯度过大 + 尺度较小 -> 克隆 (Clone): 复制自身，N 增加
  2. 梯度过大 + 尺度过大 -> 分裂 (Split): 原高斯剔除，替换为 2 个缩小版小高斯
  3. 透明度极低 -> 修剪 (Pruning): 物理删除
"""

import torch


def adaptive_density_control_step(xyz, scales, opacities, grad_norms, grad_thresh=0.5, scale_thresh=1.5):
    """
    根据给定的梯度和尺度，执行一次筛选与形态演进
    """
    print(f"--- 处理前高斯总数: {len(xyz)} ---")

    # 记录哪些高斯需要被删除、哪些新增
    keep_mask = torch.ones(len(xyz), dtype=torch.bool)

    new_xyz_list = []
    new_scales_list = []
    new_opacities_list = []

    for i in range(len(xyz)):
        grad = grad_norms[i]
        scale = scales[i]
        opacity = opacities[i]

        # 1. 垃圾修剪 (Pruning): 接近完全透明的直接丢弃
        if opacity < 0.1:
            keep_mask[i] = False
            print(f"-> 高斯 [{i}] 透明度过低 ({opacity:.2f}) -> 执行【修剪 Pruning】删除")
            continue

        # 2. 判断是否满足重构门槛 (误差梯度是否超标)
        if grad > grad_thresh:
            # 分支 A: 尺寸过大 -> 分裂 (Split)
            if scale > scale_thresh:
                keep_mask[i] = False  # 原大高斯注销
                # 分裂为两个尺寸缩小为 1/1.6 的小高斯
                new_xyz_list.append(xyz[i] + torch.tensor([0.1, 0.0, 0.0]))
                new_xyz_list.append(xyz[i] - torch.tensor([0.1, 0.0, 0.0]))
                new_scales_list.append(scale / 1.6)
                new_scales_list.append(scale / 1.6)
                new_opacities_list.append(opacity)
                new_opacities_list.append(opacity)
                print(f"-> 高斯 [{i}] 梯度大 ({grad:.2f}) 且尺寸庞大 ({scale:.2f}) -> 执行【分裂 Split】一分为二")

            # 分支 B: 尺寸较小 -> 克隆 (Clone)
            else:
                # 保留原高斯，额外克隆一个并稍微偏移
                new_xyz_list.append(xyz[i] + torch.tensor([0.05, 0.05, 0.0]))
                new_scales_list.append(scale)
                new_opacities_list.append(opacity)
                print(f"-> 高斯 [{i}] 梯度大 ({grad:.2f}) 但尺寸小巧 ({scale:.2f}) -> 执行【克隆 Clone】增加副本")

    # 整合更新后的点云张量
    survived_xyz = xyz[keep_mask]
    survived_scales = scales[keep_mask]
    survived_opacities = opacities[keep_mask]

    if new_xyz_list:
        final_xyz = torch.cat([survived_xyz, torch.stack(new_xyz_list)], dim=0)
        final_scales = torch.cat([survived_scales, torch.stack(new_scales_list)], dim=0)
        final_opacities = torch.cat([survived_opacities, torch.stack(new_opacities_list)], dim=0)
    else:
        final_xyz = survived_xyz
        final_scales = survived_scales
        final_opacities = survived_opacities

    print(f"--- 处理后高斯总数: {len(final_xyz)} ---\n")
    return final_xyz, final_scales, final_opacities


if __name__ == "__main__":
    # 模拟场景中的 3 个不同状态的高斯
    # 0号: 小巧但误差大 (应触发 Clone)
    # 1号: 庞大且误差大 (应触发 Split)
    # 2号: 几乎透明的垃圾点 (应触发 Prune)
    xyz = torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]])
    scales = torch.tensor([0.8, 2.5, 1.0])  # 尺寸大小
    opacities = torch.tensor([0.9, 0.8, 0.02])  # 不透明度
    grad_norms = torch.tensor([0.85, 0.92, 0.10])  # 梯度误差大小

    print("=== 初始高斯状态 ===")
    for i in range(len(xyz)):
        print(
            f"高斯 [{i}] | 坐标: {xyz[i].numpy()} | 尺度: {scales[i]:.2f} | 透明度: {opacities[i]:.2f} | 梯度: {grad_norms[i]:.2f}")
    print()

    # 执行密度控制调度
    final_xyz, final_scales, final_opacities = adaptive_density_control_step(
        xyz, scales, opacities, grad_norms, grad_thresh=0.5, scale_thresh=1.5
    )
    
# 0 号高斯：尺寸只有 0.8（小于 1.5），但梯度高达 0.85，触发【克隆 Clone】，原地复制出了一个新高斯；
#
# 1 号高斯：尺寸高达 2.5（大于 1.5），梯度高达 0.92，触发【分裂 Split】，原大高斯被抹除，拆成了两个尺度缩小到约 1.56 的小高斯；
#
# 2 号高斯：透明度只有 0.02，被直接识别为杂质【修剪 Prune】扔掉。
#
# 高斯总数从最开始的 3 个，精准重构为 4 个结构更合理的点。