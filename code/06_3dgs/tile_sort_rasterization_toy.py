"""
6、体验 Tile-Based 键值编码与排序机制
【模块名称】Tile-Based 键值编码与基数排序模拟 (Tile-Based Sorting Toy)
【解决痛点】消除百万像素对全量高斯的暴力遍历；通过 16x16 瓦片划分与 64 位 [Tile_ID | Depth] 键值排序，实现硬件级并行。
【核心流程】
  1. 设定 32x32 屏幕画布，划分为 4 个 16x16 的 Tile
  2. 模拟空间中分散的高斯，计算其落在哪个 Tile 以及相对于相机的深度
  3. 将 Tile ID 与 深度打包为组合键进行排序
  4. 展示各 Tile 如何按顺序直接读取属于自己的高斯列表
【张量流动】高斯中心与深度 [N, 3] -> 瓦片映射 -> 排序后的 [Tile_ID, Gaussian_ID] 执行队列
"""

import torch


def simulate_tile_assignment_and_sort(gaussians_2d, depths, tile_size=16, canvas_w=32, canvas_h=32):
    """
    极简模拟 3DGS CUDA 光栅化器的前期核心流程：Tile 映射与深度排序
    """
    tiles_x = canvas_w // tile_size
    tiles_y = canvas_h // tile_size

    key_list = []

    print(
        f"画布大小: {canvas_w}x{canvas_h} | Tile 大小: {tile_size}x{tile_size} | 瓦片网格: {tiles_x}x{tiles_y} (共 {tiles_x * tiles_y} 个 Tile)")

    for g_id, (pos, depth) in enumerate(zip(gaussians_2d, depths)):
        # 计算该高斯中心点落在哪个 Tile 内
        tile_x = int(pos[0] // tile_size)
        tile_y = int(pos[1] // tile_size)

        # 边界防溢出
        if 0 <= tile_x < tiles_x and 0 <= tile_y < tiles_y:
            tile_id = tile_y * tiles_x + tile_x

            # ==========================================================
            # 【重点解剖行 1】：构造 [Tile ID, 深度, 高斯编号] 复合键
            # 真实 CUDA 中使用 64位整数: (tile_id << 32) | depth_bits
            # ==========================================================
            key_list.append({
                "tile_id": tile_id,
                "depth": depth.item(),
                "gaussian_id": g_id
            })

    # ==========================================================
    # 【重点解剖行 2】：双重排序 (先按 Tile ID 聚集，同 Tile 内严格按深度升序)
    # ==========================================================
    sorted_keys = sorted(key_list, key=lambda item: (item["tile_id"], item["depth"]))

    return sorted_keys


if __name__ == "__main__":
    # 模拟 4 个不同位置、不同深度的高斯
    # 画布大小 32x32，分为 4 个 Tile (每个 16x16)：
    # Tile 0: x in [0,16), y in [0,16)   |  Tile 1: x in [16,32), y in [0,16)
    # Tile 2: x in [0,16), y in [16,32)  |  Tile 3: x in [16,32), y in [16,32)

    gaussians_pos = torch.tensor([
        [5.0, 5.0],  # 高斯 0: 落在 Tile 0，距离深度 8.5 (较远)
        [20.0, 8.0],  # 高斯 1: 落在 Tile 1，距离深度 3.2
        [10.0, 10.0],  # 高斯 2: 落在 Tile 0，距离深度 2.1 (较近)
        [25.0, 25.0]  # 高斯 3: 落在 Tile 3，距离深度 5.0
    ])
    depths = torch.tensor([8.5, 3.2, 2.1, 5.0])

    print("--- 原始输入高斯点 ---")
    for i in range(len(gaussians_pos)):
        print(f"高斯 [{i}] | 屏幕坐标: {gaussians_pos[i].numpy()} | 相机深度: {depths[i].item():.1f}")
    print()

    # 执行模拟分块与排序
    sorted_queue = simulate_tile_assignment_and_sort(gaussians_pos, depths)

    print("\n--- GPU 排序后得到的执行指令队列 (各 Tile 无缝流式并发) ---")
    for item in sorted_queue:
        print(f"Tile [{item['tile_id']}] 准备渲染 -> 高斯 [{item['gaussian_id']}] | 深度: {item['depth']:.1f}")