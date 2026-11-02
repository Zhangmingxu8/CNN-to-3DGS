"""
5、
【模块名称】潜空间扩散机制 (Latent Diffusion / Stable Diffusion Toy)
【解决痛点】像素级扩散计算量过大；使用预训练 VAE 将图像下采样 8 倍压缩到潜空间，使 UNet 仅在低维 Latent 上做扩散计算。
【核心流程】
  1. 模拟输入一张 512x512 的高清 RGB 图像
  2. 使用 VAE Encoder（编码器） 空间压缩 8 倍，得到 [B, 4, 64, 64] 的潜变量 z_0
  3. 在潜空间上进行正向加噪与 UNet 去噪
  4. 使用 VAE Decoder（解码器） 将去噪后的潜变量还原为 512x512 的像素图像
【张量流动】原图 [1, 3, 512, 512] -> VAE 压缩 -> [1, 4, 64, 64] -> UNet 潜空间计算 -> VAE 解压 -> 原图 [1, 3, 512, 512]
"""

import torch
import torch.nn as nn


# 1. 模拟 VAE 自编码器 (提供 8 倍空间压缩与还原能力)
class ToyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器：连续通过 stride=2 卷积三次，长宽缩小 2*2*2 = 8 倍 (512 -> 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # 512 -> 256 输出尺寸有单独的计算公式
            nn.ReLU(),#引入非线性表达能力
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 4, kernel_size=4, stride=2, padding=1),  # 128 -> 64 (通道压缩为 4)
        )

        # 解码器：连续三次转置卷积，尺寸放大 8 倍 (64 -> 512)
        self.decoder = nn.Sequential(
            #反卷积
            nn.ConvTranspose2d(4, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # 256 -> 512 (还原 RGB 3通道)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# 2. 模拟潜空间内的 UNet (输入输出均是在 64x64 潜空间中)
class LatentUNetToy(nn.Module):
    def __init__(self):
        super().__init__()
        # 仅针对 4 通道的 Latent 进行特征提取
        self.block = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 4, kernel_size=3, padding=1)
        )

    def forward(self, z_t):
        return self.block(z_t)


if __name__ == "__main__":
    # 实例化网络
    vae = ToyVAE()
    latent_unet = LatentUNetToy()
    vae.eval()
    latent_unet.eval()

    # 1. 模拟输入一张 512x512 的大图 (RGB 3通道)
    pixel_image = torch.randn(1, 3, 512, 512)
    print("【阶段 1：原始像素空间】输入高清图像形状:", pixel_image.shape)

    # ==========================================================
    # 【重点解剖行 1】：VAE 压缩入潜空间 (空间尺寸缩小 8 倍)
    # ==========================================================
    with torch.no_grad():
        z_0 = vae.encode(pixel_image)
    print("【阶段 2：潜空间压缩】压缩后 Latent 特征形状:", z_0.shape)

    # 2. 在潜空间内进行操作 (加噪与 UNet 预测)
    latent_noise = torch.randn_like(z_0)
    z_t = 0.7 * z_0 + 0.3 * latent_noise  # 模拟在潜空间加噪

    # ==========================================================
    # 【重点解剖行 2】：UNet 在极小尺寸的潜空间上高效预测噪声
    # ==========================================================
    pred_latent_noise = latent_unet(z_t)
    print("【阶段 3：潜空间计算】UNet 预测的潜噪声形状:", pred_latent_noise.shape)

    # 模拟去噪扣除噪声后的干净潜向量
    z_clean = z_t - 0.3 * pred_latent_noise

    # ==========================================================
    # 【重点解剖行 3】：VAE 解码还原回高清彩色图像
    # ==========================================================
    with torch.no_grad():
        reconstructed_image = vae.decode(z_clean)
    print("【阶段 4：像素空间还原】解码还原后图像形状:", reconstructed_image.shape)

#
# 四、 本次代码的【核心观察点】
# 在 PyCharm 中运行该脚本，重点盯紧控制台打印的四个形状阶段：
#
# 压缩前：[1, 3, 512, 512]（像素总数近 80 万）；
#
# 压缩后（Latent）：[1, 4, 64, 64]（数值总数仅 1.6 万，体积暴跌到原来的 2%）；
#
# 主干计算：UNet 从头到尾都在对 [1, 4, 64, 64] 做运算，完全不碰 512 的大图；
#
# 最终还原：只有当所有去噪步数全跑完之后，才用 VAE Decoder 只做一次解压放大回 [1, 3, 512, 512]。