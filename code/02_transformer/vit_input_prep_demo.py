#3、说明Transformer 是如何解决没有方位感以及在输出的很多块中选取哪个块去做全连接层的图像识别？

import torch
import torch.nn as nn

# 模拟参数配置
batch_size = 1
embed_dim = 64
num_patches = 16  # 28x28 切成 7x7，得到 16 个图块

# 1. 模拟 Patch Embedding 出来的切片特征
# 形状: [1, 16, 64]
patches = torch.randn(batch_size, num_patches, embed_dim)
print(f"1. 原始图块特征形状: {patches.shape}")

# 2. 定义可学习的 [CLS] Token (班长)
# 形状: [1, 1, 64]
cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

# 把班长拼在 16 个图块的最前面 (在第 1 维拼接)
# [1, 1, 64] 拼上 [1, 16, 64] -> 变成 [1, 17, 64]
cls_tokens = cls_token.expand(batch_size, -1, -1)
x = torch.cat((cls_tokens, patches), dim=1)
print(f"2. 加上 [CLS] Token 后的序列形状: {x.shape}")

# 3. 定义可学习的位置编码 Positional Embedding (17 个座位的编号)
# 形状: [1, 17, 64]
pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

# 4. 特征与位置编码直接逐元素相加！
x = x + pos_embed
print(f"3. 加上位置编码后的最终完整输入形状: {x.shape}")


#
# 你会看到张量从 [1, 16, 64] 变成了 [1, 17, 64]。
# 这 17 个 Token，就是兼备了语义信息、空间坐标与全局联络代理（CLS）的完整工业级输入，可以直接送进我们在上一节写的 Self-Attention 模块！