"""
2、 扩散训练（神经网络训练）
【模块名称】极简 UNet 噪声预测与单步训练 (UNet Denoise Toy)
【解决痛点】直观理解扩散模型中的神经网络是如何接收加噪图并预测出同尺寸噪声的。
【核心流程】
  1. 采样真实噪声 noise，用公式加噪得到 x_t
  2. 将 x_t 与时间步 t 喂给 UNet
  3. UNet 经过 下采样（缩小尺寸，看全图） -> 瓶颈层 -> 上采样(放大尺寸 看细节，带跳跃连接) 输出预测噪声
  4. 计算真实噪声与预测噪声的 MSE Loss 并反向传播更新网络
【张量流动】加噪图 [1, 1, 28, 28] -> UNet -> 预测噪声 [1, 1, 28, 28] -> 与真实噪声算 Loss
"""

import torch
import torch.nn as nn
import torch.optim as optim


class ToyUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 1. 编码器 (下采样): 提取高层轮廓
        self.down1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # 28x28 -> 14x14

        # 2. 瓶颈层 (最深处)
        self.bottleneck = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # 3. 解码器 (上采样): 还原尺寸并融合细节
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  # 14x14 -> 28x28
        # 注意：这里输入通道是 16 (上采样后) + 16 (来自下采样跳跃连接拼过来的通道) = 32
        self.up_conv = nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 步骤 A: 下采样并保存浅层特征 (用于跳跃连接)
        feat1 = self.relu(self.down1(x))  # 形状: [B, 16, 28, 28]
        p1 = self.pool(feat1)  # 形状: [B, 16, 14, 14]

        # 步骤 B: 瓶颈层处理
        b = self.relu(self.bottleneck(p1))  # 形状: [B, 32, 14, 14]

        # 步骤 C: 上采样并执行【跳跃连接 (Skip Connection)】
        up = self.upsample(b)  # 放大回: [B, 32, 28, 28]
        # ==========================================================
        # 【重点解剖行 1】：横向拼接浅层细节（跳跃连接）与深层语义！
        # ==========================================================
        cat_feat = torch.cat([up, feat1], dim=1)  # 形状拼成: [B, 48, 28, 28]
        #观察上采样后的 up（32 通道）和最开始保存的 feat1（16 通道）在通道维度上直接拼在一起变成了 48 通道。这就是 UNet 能够精准勾勒细节的最核心特征。

        # 步骤 D: 卷积还原为单通道噪声图
        out = self.relu(self.up_conv(cat_feat))
        pred_noise = self.out_conv(out)  # 形状还原为: [B, 1, 28, 28]
        return pred_noise


if __name__ == "__main__":
    # 1. 模拟输入数据
    x_0 = torch.randn(1, 1, 28, 28)  # 原始干净图
    true_noise = torch.randn_like(x_0)  # 人类随机添加的真实高斯胡椒面 (标准答案)

    # 模拟在某个时间步 t 加噪后的图片 (原图与噪声混合)
    x_t = 0.5 * x_0 + 0.5 * true_noise  # 加噪图，送进网络去猜

    # 2. 初始化网络与优化器
    model = ToyUNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("--- 开始训练 UNet 识别并预测噪声 ---")
    for step in range(5):
        optimizer.zero_grad()

        # 让 UNet 猜测噪声
        pred_noise = model(x_t)

        # ==========================================================
        # 【重点解剖行 2】：计算真实噪声与预测噪声的差距并求导！
        # ==========================================================
        loss = criterion(pred_noise, true_noise) #做均方差MSE,就是把 UNet 猜出来的噪声像素 和 我们一开始扔进去的真实噪声像素，一个像素一个像素对应相减、求平方、再算平均值。
        #如果猜的准的话，MSE的值会很小。如果网络猜的不准，MSE会比较大


        #反向求导步骤。
        #算出来的loss有大有小，根据loss的值要反过来对神经网络里面的参数进行调节
        #这个时候导数的值就会反应参数调整是否合理
        #导数为+：说明调大参数会导致误差变大，需要调小参数
        #导数为-：说明调大参数会导致误差变小，需要调大参数
        loss.backward()


        optimizer.step()

        print(f"训练步骤 [{step + 1}/5] | 噪声预测 MSE Loss: {loss.item():.4f}")

    print("\n训练验证完成！UNet 输出的形状完全对齐输入尺寸:", pred_noise.shape)