# 5、

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math


# 1. 搬运上一节写好的两个核心积木
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10, embed_dim=64, depth=2):
        super().__init__()
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


# 2. 准备数据 (取前 2000 张图跑快速验证)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
train_subset = torch.utils.data.Subset(train_dataset, range(2000))
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

# 3. 初始化模型、损失函数与优化器
device = torch.device("cpu")
model = SimpleViT(img_size=28, patch_size=7, in_channels=1, num_classes=10, embed_dim=64, depth=2).to(device)
criterion = nn.CrossEntropyLoss()
# 提示：ViT 通常配合 AdamW 优化器，学习率设小一点 (比如 0.001)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# 4. 跑 2 个 Epoch 观察 Loss 下降
print("开始训练从零手写的 SimpleViT...")
for epoch in range(2):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/2], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

print("\n训练完成！纯手写的 ViT 在真实图像数据上完成了反向传播更新与收敛。")