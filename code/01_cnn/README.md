# 01 · CNN 实践

对应笔记：[`docs/01-cnn.md`](../../docs/01-cnn.md)

## 文件

| 文件 | 作用 | 关键观察 |
| :--- | :--- | :--- |
| `train_cnn_demo.py` | 从零训练一个 2 层卷积的 Fashion-MNIST 分类器 | 张量 `[B,1,28,28] → [B,16,14,14] → [B,32,7,7] → [B,1568] → [B,10]` |
| `predict_demo.py` | 加载权重，对单张图推理并可视化 | 单张图也要 `unsqueeze(0)` 补 batch 维 |
| `resblock_demo.py` | 残差块的两种情形：恒等 / 降采样投影 | 通道不同时必须用 1×1 卷积对齐 |
| `transfer_learning_demo.py` | 冻结 ResNet-18 主干，只训练分类头 | 可训练参数只剩最后一层 |

## 运行

```bash
python code/01_cnn/resblock_demo.py            # 不需要数据，秒出结果
python code/01_cnn/train_cnn_demo.py           # 首次运行自动下载 Fashion-MNIST（约 30MB）
python code/01_cnn/predict_demo.py             # 依赖上一步生成的 fashion_cnn.pth
python code/01_cnn/transfer_learning_demo.py   # 首次运行下载 ResNet-18 预训练权重（约 45MB）
```

## 预期输出

```text
测试 1 - 恒等残差块输出形状: torch.Size([1, 16, 28, 28])
测试 2 - 降采样残差块输出形状: torch.Size([1, 32, 14, 14])

Epoch [1/2], Batch [200/938], Loss: 0.6...
...
模型在 10,000 张测试图片上的准确率为: 89.xx%
```

> 2 个 epoch 只是"跑通验证"，不是调优结果。想更高可以加 epoch、数据增强或换更强的主干。

## 注意

- `data/`、`fashion_cnn.pth` 已被 `.gitignore` 忽略，不会上传；
- 全部脚本默认 CPU，几秒钟一个 epoch。
