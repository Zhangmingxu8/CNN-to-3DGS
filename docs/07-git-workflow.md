# 07 · Git 工作流：让每一次修改都可回溯

> 这篇不是 Git 教程，而是**这个仓库真正用到的操作清单**。
> 学习笔记最大的价值是"半年后还能找到当时的自己"，而 Git 就是那台时光机。

---

## 1. 从零到第一次 push

```bash
# 1. 本地建仓
mkdir CNN-to-3DGS && cd CNN-to-3DGS
git init

# 2. 写代码 / 写笔记
# ...

# 3. 暂存 + 提交
git add .
git commit -m "feat: 初始化仓库，完成 CNN 模块笔记与 4 个 Demo"

# 4. 绑定远程仓库（SSH 方式）
git remote add origin git@github.com:Zhangmingxu8/CNN-to-3DGS.git
git branch -M main
git push -u origin main
```

> 首次使用 SSH 需要生成密钥并把公钥贴到 GitHub → Settings → SSH and GPG keys。
> 验证：`ssh -T git@github.com`，看到 `Hi <username>!` 即成功。

---

## 2. 日常循环：只需要三条命令

```bash
git add .
git commit -m "docs: 补充 3DGS 球谐函数推导"
git push
```

配置好 SSH 与 remote 之后，**再也不需要重复配置**。

---

## 3. .gitignore：什么不该上传

| 类型 | 例子 | 原因 |
| :--- | :--- | :--- |
| 缓存 | `__pycache__/`、`.pytest_cache/` | 可再生成 |
| IDE | `.idea/`、`.vscode/` | 个人环境配置 |
| 数据 | `data/`、`datasets/` | 体积大，脚本会自动下载 |
| 权重 | `*.pth`、`*.ckpt` | 体积大，且可重新训练 |
| 输出 | `runs/`、`logs/`、`outputs/` | 训练产物 |
| 系统 | `.DS_Store`、`Thumbs.db` | 平台垃圾文件 |

```gitignore
__pycache__/
.idea/
.vscode/
data/
*.pth
runs/
```

> 判断标准：**别人 clone 下来能不能自己生成？** 能，就不要提交。

---

## 4. Commit Message 规范

采用简化版 Conventional Commits：

| 前缀 | 含义 | 示例 |
| :--- | :--- | :--- |
| `feat` | 新增功能 / 模块 | `feat: 新增 3DGS 球谐函数 Demo` |
| `docs` | 只改文档 | `docs: 补充 LoRA 参数量推导` |
| `fix` | 修 bug | `fix: 修正残差块投影卷积的 stride` |
| `refactor` | 重构（不改行为） | `refactor: 统一所有脚本的文件头注释` |
| `chore` | 杂项 | `chore: 更新 requirements.txt` |

原则：**一次提交只做一件事**。这样 `git log` 本身就是一份学习时间线。

---

## 5. 常用排错

| 问题 | 原因 | 解决 |
| :--- | :--- | :--- |
| `Permission denied (publickey)` | SSH key 未绑定 | `ssh -T git@github.com` 验证，重新贴公钥 |
| `Updates were rejected` | 远程有新提交 | `git pull --rebase` 后再 push |
| 误提交了大文件 | 忘记写 .gitignore | `git rm --cached <file>` 后重新提交 |
| 想撤销上一次提交（未 push） | — | `git reset --soft HEAD~1` |
| 想看某次改动 | — | `git show <commit>` / `git log -p` |

---

## 6. 这个仓库的实践

- 每个模块一个目录，提交信息按模块划分，`git log` 能看出学习顺序；
- 数据集、权重、缓存全部 ignore，仓库体积保持在几 MB；
- README + docs 与代码同步更新，保证"文档描述的 = 代码实际做的"。
