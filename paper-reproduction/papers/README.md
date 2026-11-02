# papers/ · 论文目录

每篇论文一个文件夹，命名规则：

```text
YYYY-FirstAuthor-ShortName/
```

示例：

```text
papers/
├── 2026-Bao-DCx/          # SIGGRAPH 2026，已收录（精读完成，复现进行中）
│   ├── README.md          # 论文卡片
│   ├── notes.md           # 精读笔记
│   ├── glossary.md        # 术语表
│   ├── code-map.md        # 代码地图 + 学习计划
│   ├── repro.md           # 复现记录
│   ├── study-log.md       # 学习日志
│   └── assets/            # 论文插图（CC BY 4.0）
└── 2024-XXXX-YYYY/
```

## 新增一篇的步骤

```bash
mkdir -p paper-reproduction/papers/2026-Bao-DCx/assets
cp paper-reproduction/templates/paper-card.md  paper-reproduction/papers/2026-Bao-DCx/README.md
cp paper-reproduction/templates/paper-note.md  paper-reproduction/papers/2026-Bao-DCx/notes.md
cp paper-reproduction/templates/repro-log.md   paper-reproduction/papers/2026-Bao-DCx/repro.md
```

然后在 [`../README.md`](../README.md) 的状态看板里加一行。

> 实际情况比模板多两个文件：`glossary.md`（术语表，初学时很有用）和 `study-log.md`（学习日志）。

## 约定

- **一篇论文一个目录**，不要按会议或方向分，避免后期移动；
- `assets/` 里的图命名要有信息量：`overview.png`、`ablation-depth.png`、`failure-case.png`；
- 官方代码不要整包提交，记录 `repo @ commit`，需要改动时导出 `.patch` 放 `code/`。
