# papers/ · 论文目录

每篇论文一个文件夹，命名规则：

```text
YYYY-FirstAuthor-ShortName/
```

示例：

```text
papers/
├── 2023-Kerbl-3DGS/
│   ├── README.md      # 论文卡片（元信息 + 三句话总结）
│   ├── notes.md       # 精读笔记
│   ├── repro.md       # 复现记录
│   ├── assets/        # 架构图、结果对比图、训练曲线
│   └── code/          # 复现补丁 / 配置（不放大仓库）
└── 2024-XXXX-YYYY/
```

## 新增一篇的步骤

```bash
mkdir -p paper-reproduction/papers/2023-Kerbl-3DGS/assets
cp paper-reproduction/templates/paper-card.md  paper-reproduction/papers/2023-Kerbl-3DGS/README.md
cp paper-reproduction/templates/paper-note.md  paper-reproduction/papers/2023-Kerbl-3DGS/notes.md
cp paper-reproduction/templates/repro-log.md   paper-reproduction/papers/2023-Kerbl-3DGS/repro.md
```

然后在 [`../README.md`](../README.md) 的状态看板里加一行。

## 约定

- **一篇论文一个目录**，不要按会议或方向分，避免后期移动；
- `assets/` 里的图命名要有信息量：`overview.png`、`ablation-depth.png`、`failure-case.png`；
- 官方代码不要整包提交，记录 `repo @ commit`，需要改动时导出 `.patch` 放 `code/`。
