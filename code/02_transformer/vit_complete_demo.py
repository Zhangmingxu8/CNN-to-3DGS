#4、跑vit全程

import torch
import torch.nn as nn
import math

# ===============================
# 1. 自注意力模块 (Self-Attention)
# ===============================
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # x: [Batch, Seq_Len, Embed_Dim]
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 相似度打分矩阵: [Batch, Seq_Len, Seq_Len]
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权聚合 Value: [Batch, Seq_Len, Embed_Dim]
        out = torch.matmul(attn_weights, V)
        return out


# ==========================================
# 2. Transformer 编码块 (带残差连接与前馈网络)
# ==========================================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        # 前馈网络 MLP: 升维再降维，做特征的深度非线性提炼
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )

    def forward(self, x):
        # 残差连接 1: x + Attention(x)
        x = x + self.attn(self.norm1(x))
        # 残差连接 2: x + MLP(x)
        x = x + self.mlp(self.norm2(x))
        return x


# ================================
# 3. 完整的 Vision Transformer (ViT)
# ================================
class SimpleViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10, embed_dim=64, depth=2):
        super().__init__()

        # 1) Patch Embedding 模块
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 2) [CLS] Token 与 位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # 3) 堆叠多个 Transformer 编码块 (比如堆 2 层)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        # 4) 分类头: 接收班长特征 (64维)，输出类别得分 (10类)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # 步骤 1: 切片映射 [B, 1, 28, 28] -> [B, 64, 4, 4] -> [B, 16, 64]
        x = self.proj(x).flatten(2).transpose(1, 2)

        # 步骤 2: 拼装班长与位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # 变成 [B, 17, 64]
        x = x + self.pos_embed  # 逐元素叠加空间坐标

        # 步骤 3: 穿越 Transformer 编码块 (全局打分与特征融合)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # 步骤 4: 仅取出第 0 个位置的 [CLS] Token (班长) 进行分类，只取出大总管进行全连接识别
        cls_out = x[:, 0]  # [B, 64]
        logits = self.head(cls_out)  # [B, 10]
        return logits


# ================================
# 4. 模拟测试：前向传播
# ================================
if __name__ == "__main__":
    # 创建模型实例
    model = SimpleViT(img_size=28, patch_size=7, in_channels=1, num_classes=10, embed_dim=64, depth=2)

    # 模拟一张 Fashion-MNIST 输入图: [Batch=1, C=1, H=28, W=28]
    dummy_img = torch.randn(1, 1, 28, 28)

    output = model(dummy_img)
    print("模型推理成功！")
    print(f"最终输出预测得分形状 [Batch, Num_Classes]: {output.shape}")
    print("10 个类别的非归一化预测得分 (Logits):\n", output)