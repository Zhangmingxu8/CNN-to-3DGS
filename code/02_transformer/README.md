# 02 · Transformer & ViT 实践

对应笔记：[`docs/02-transformer.md`](../../docs/02-transformer.md)

## 文件（按学习顺序）

| 文件 | 作用 | 张量变化 |
| :--- | :--- | :--- |
| `patch_embed_demo.py` | 用 stride=patch 的卷积完成切片 + 映射 | `[B,1,28,28] → [B,16,64]` |
| `vit_input_prep_demo.py` | 拼接 CLS Token 与位置编码 | `[B,16,64] → [B,17,64]` |
| `self_attention_demo.py` | 最小自注意力：QKV → 打分 → softmax → 聚合 | `[B,16,64] → [B,16,16] → [B,16,64]` |
| `mha_demo.py` | 多头注意力：切分 / 并行 / 拼接 | `[B,17,64] → [B,4,17,16] → [B,17,64]` |
| `norm_comparison_demo.py` | LayerNorm vs BatchNorm 的统计维度 | 每个 token 内部均值≈0 |
| `causal_attention_demo.py` | 因果掩码，禁止偷看未来 | 上三角权重全为 0 |
| `vit_complete_demo.py` | 组装完整 ViT 并做一次前向 | `[1,1,28,28] → [1,10]` |
| `train_vit_demo.py` | 在 2000 张 Fashion-MNIST 上训练 | loss 逐步下降 |

## 运行

```bash
python code/02_transformer/patch_embed_demo.py
python code/02_transformer/mha_demo.py
python code/02_transformer/vit_complete_demo.py
python code/02_transformer/train_vit_demo.py     # 自动下载数据
```

## 注意

- ViT 没有卷积的归纳偏置，小数据上容易过拟合；`train_vit_demo.py` 只取 2000 张图，目的是验证反向传播链路，不追求精度；
- `transpose` 后若要 `view`，记得先 `.contiguous()`。
