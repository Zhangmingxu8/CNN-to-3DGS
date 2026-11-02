# 04 · Diffusion 实践

对应笔记：[`docs/04-diffusion.md`](../../docs/04-diffusion.md)

## 文件

| 文件 | 作用 | 观察点 |
| :--- | :--- | :--- |
| `forward_diffusion_demo.py` | 闭式解一步加噪 | t=0 与 t=999 的信号/噪声比例 |
| `unet_denoise_toy.py` | 极简 UNet 预测噪声 + MSE 训练 | 跳跃连接是 `torch.cat` |
| `diffusion_sampling_demo.py` | DDPM 反向采样循环 | 逐步去噪时均值/方差的变化 |
| `cfg_condition_demo.py` | 无分类器引导（CFG） | `ε_u + s(ε_c − ε_u)` |
| `latent_diffusion_toy.py` | VAE 压缩 + 潜空间扩散 | `[1,3,512,512] → [1,4,64,64]` |

## 运行

```bash
python code/04_diffusion/forward_diffusion_demo.py
python code/04_diffusion/unet_denoise_toy.py
python code/04_diffusion/diffusion_sampling_demo.py
python code/04_diffusion/cfg_condition_demo.py
python code/04_diffusion/latent_diffusion_toy.py
```

## 预期输出

```text
时间步 t =   0 | 原图保留比例: 0.9999 | 噪声掺入比例: 0.0105
时间步 t = 500 | 原图保留比例: 0.5047 | 噪声掺入比例: 0.8633
时间步 t = 999 | 原图保留比例: 0.0064 | 噪声掺入比例: 1.0000
```

## ⚠️ 重要说明

`diffusion_sampling_demo.py` 中的 UNet 是**随机初始化、未训练**的网络，
它演示的是采样循环的数学行为，最终输出仍然接近噪声，**不是一张有意义的图像**。
这是刻意设计，用来观察"每一步均值如何变化"，而不是生成质量。
