"""
生成 README / 笔记中引用的全部插图。

依赖：numpy、matplotlib（见 requirements.txt）。
数据：优先读取 code/01_cnn/data/FashionMNIST；若不存在则用合成图像代替。

用法：
    python tools/make_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 全局样式
INK = "#1f2328"
MUTED = "#6e7781"
GRID = "#e6e8eb"
C0, C1, C2, C3, C4, C5 = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"

for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    try:
        fm.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [ok] {path.relative_to(ROOT)}")


# ---------------------------------------------------------------- 数据加载
def load_fashion(n: int = 1):
    """返回若干张 28x28 的 Fashion-MNIST 图像（0-1），失败则用合成图。"""
    raw = ROOT / "code" / "01_cnn" / "data" / "FashionMNIST" / "raw"
    img_path = raw / "train-images-idx3-ubyte"
    lbl_path = raw / "train-labels-idx1-ubyte"
    if img_path.exists() and lbl_path.exists():
        with open(img_path, "rb") as f:
            f.seek(16)
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
        with open(lbl_path, "rb") as f:
            f.seek(8)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        idx = np.where(labels == 9)[0][:n]  # 9 = ankle boot，轮廓清晰
        if len(idx):
            return data[idx].astype(np.float32) / 255.0

    # 合成图：鞋子形状的简单剪影
    yy, xx = np.mgrid[0:28, 0:28]
    img = np.zeros((n, 28, 28), dtype=np.float32)
    img[:, 6:20, 8:20] = 1.0
    img[:, 16:22, 4:24] = 1.0
    img = np.clip(img + 0.15 * np.exp(-((yy - 14) ** 2 + (xx - 14) ** 2) / 60.0), 0, 1)
    return img


# ---------------------------------------------------------------- 1. 路线图
def fig_roadmap() -> None:
    modules = [
        ("06", "3D Gaussian Splatting", "10.30 – 11.01", C5),
        ("05", "Flow Matching", "10.29 – 10.30", C4),
        ("04", "Diffusion", "10.28 – 10.29", C3),
        ("03", "LLM & Agent", "10.27 – 10.29", C2),
        ("02", "Transformer & ViT", "10.07 – 10.27", C1),
        ("01", "CNN", "09.28 – 10.07", C0),
    ]
    start = np.datetime64("2026-09-28")
    fig, ax = plt.subplots(figsize=(13.5, 5.4))

    for i, (num, name, span, color) in enumerate(modules):
        a, b = span.split(" – ")
        d0 = np.datetime64(f"2026-{a.replace('.', '-')}")
        d1 = np.datetime64(f"2026-{b.replace('.', '-')}")
        x0 = (d0 - start).astype(int)
        width = max((d1 - d0).astype(int), 1)
        ax.barh(i, width, left=x0, height=0.52, color=color, alpha=0.9,
                edgecolor="white", linewidth=1.6, zorder=3)
        ax.text(x0 + width + 0.6, i, f"{span}", va="center", ha="left",
                fontsize=10.5, color=MUTED, zorder=4)
        ax.text(-0.6, i, f"{num}  {name}", va="center", ha="right",
                fontsize=11.5, color=INK, fontweight="bold")

    ax.set_xlim(-0.5, 45)
    ax.set_ylim(-0.7, 5.7)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 36, 7))
    ax.set_xticklabels(["09.28", "10.05", "10.12", "10.19", "10.26", "11.02"], fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.012, 0.975, "五周走完六个模块：从卷积到三维高斯泼溅",
             fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.915, "计划 3 个月 · 实际 5 周（2026.09.28 – 11.01） · 30 个可运行 Demo",
             fontsize=10.5, color=MUTED, va="top")
    fig.subplots_adjust(top=0.80)
    save(fig, "roadmap.png")


# ---------------------------------------------------------------- 2. CNN 管线
def _box(ax, x, y, w, h, title, sub, color, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.4, edgecolor=color, facecolor=color + "18"))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
            fontsize=8.4, color=MUTED)


def fig_cnn_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 3.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        ("输入", "[B, 1, 28, 28]", C0),
        ("Conv 3×3 · 16", "[B, 16, 28, 28]", C1),
        ("MaxPool 2×2", "[B, 16, 14, 14]", C1),
        ("Conv 3×3 · 32", "[B, 32, 14, 14]", C2),
        ("MaxPool 2×2", "[B, 32, 7, 7]", C2),
        ("Flatten", "[B, 1568]", C4),
        ("FC + ReLU", "[B, 128]", C3),
        ("FC", "[B, 10]", C5),
    ]
    w, h, gap = 1.35, 0.95, 0.30
    for i, (t, s, c) in enumerate(boxes):
        x = 0.15 + i * (w + gap)
        _box(ax, x, 1.0, w, h, t, s, c)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.47), (x + w + gap - 0.03, 1.47),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color="#9aa4ae", linewidth=1.3))
    ax.text(0.15, 0.45, "通道数 ↑ 空间尺寸 ↓：网络逐步把「像素」抽象成「语义」",
            fontsize=10, color=MUTED)
    ax.text(0.15, 2.45, "一张 28×28 灰度图的完整数据流", fontsize=14,
            fontweight="bold", color=INK)
    save(fig, "cnn_pipeline.png")


# ---------------------------------------------------------------- 3. ViT patch
def fig_vit_patches() -> None:
    img = load_fashion(1)[0]
    fig = plt.figure(figsize=(12.6, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.05], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    for k in range(1, 4):
        ax.axhline(k * 7 - 0.5, color="#C44E52", lw=1.2, alpha=0.9)
        ax.axvline(k * 7 - 0.5, color="#C44E52", lw=1.2, alpha=0.9)
    for j in range(4):
        for i in range(4):
            ax.text(i * 7 + 3.5, j * 7 + 3.5, str(j * 4 + i), color="#C44E52",
                    fontsize=8.5, ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("28×28 → 16 个 7×7 图块", fontsize=12, color=INK, pad=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-0.6, 17.4)
    ax2.set_ylim(-1.4, 3.9)
    ax2.axis("off")

    # CLS token
    ax2.add_patch(FancyBboxPatch((0.0, 0.9), 0.85, 0.85, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor=C3 + "22", edgecolor=C3, lw=1.3))
    ax2.text(0.42, 1.33, "CLS", ha="center", va="center", fontsize=8.6, color=C3, fontweight="bold")

    for i in range(16):
        x = 1.15 + i * 0.98
        j, k = divmod(i, 4)
        ax2.add_patch(FancyBboxPatch((x, 0.9), 0.82, 0.82, boxstyle="round,pad=0.02,rounding_size=0.06",
                                     facecolor="#f6f8fa", edgecolor="#d0d7de", lw=1.0))
        sub = img[j * 7:(j + 1) * 7, k * 7:(k + 1) * 7]
        inset = ax2.inset_axes([x + 0.14, 0.9 + 0.14, 0.54, 0.54], transform=ax2.transData)
        inset.imshow(sub, cmap="gray", vmin=0, vmax=1)
        inset.set_xticks([]); inset.set_yticks([])
        for s in inset.spines.values():
            s.set_visible(False)
        ax2.text(x + 0.41, 0.70, f"#{i}", ha="center", va="center", fontsize=7.4, color=MUTED)

    ax2.annotate("", xy=(17.1, 1.32), xytext=(1.0, 1.32),
                 arrowprops=dict(arrowstyle="-|>", color="#9aa4ae", lw=1.2))
    ax2.text(9.0, 2.05, "拼接为序列：[B, 17, 64]（16 个图块 + 1 个 CLS）",
             ha="center", va="bottom", fontsize=10, color=INK)
    ax2.text(0.0, 3.25, "每个图块被线性映射成 64 维语义向量", fontsize=12, color=INK)
    ax2.text(0.0, -0.55, "自注意力让 17 个 token 两两交互 → 第一层就拥有全局感受野",
             fontsize=9.6, color=MUTED)
    save(fig, "vit_patches.png")


# ---------------------------------------------------------------- 4. 前向加噪
def fig_diffusion_forward() -> None:
    x0 = load_fashion(1)[0] * 2.0 - 1.0
    T = 1000
    betas = np.linspace(1e-4, 0.02, T)
    abar = np.cumprod(1.0 - betas)

    rng = np.random.default_rng(7)
    eps = rng.standard_normal(x0.shape)
    steps = [0, 250, 500, 750, 999]

    fig, axes = plt.subplots(1, len(steps), figsize=(13.2, 3.35))
    for ax, t in zip(axes, steps):
        xt = np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps
        ax.imshow(xt, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(f"t = {t}", fontsize=11.5, color=INK, pad=6)
        ax.set_xlabel(f"信号 $\\sqrt{{\\bar\\alpha_t}}$ = {np.sqrt(abar[t]):.2f}", fontsize=9, color=MUTED, labelpad=5)
    fig.suptitle("正向加噪：闭式解一步到位，无需循环 1000 次", fontsize=14,
                 fontweight="bold", x=0.012, ha="left", y=1.06)
    save(fig, "diffusion_forward.png")


# ---------------------------------------------------------------- 5. Flow Matching
def _velocity(x, t):
    return -2.0 * x * t + np.sin(x)


def _integrate(x, t0, t1, n, method):
    dt = (t1 - t0) / n
    t = t0
    for _ in range(n):
        if method == "euler":
            x = x + _velocity(x, t) * dt
        else:
            v1 = _velocity(x, t)
            x_mid = x + v1 * dt / 2
            x = x + _velocity(x_mid, t + dt / 2) * dt
        t += dt
    return x


def fig_flow() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.7),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- 左：直线轨迹
    rng = np.random.default_rng(3)
    n = 240
    src = rng.normal(0, 0.45, (n, 2))
    theta = rng.uniform(0, 2 * np.pi, n)
    r = 1.9 + rng.normal(0, 0.12, n)
    tgt = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    for i in range(0, n, 3):
        ax1.plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1]],
                 color=C0, alpha=0.22, lw=1.0, zorder=1)
    ax1.scatter(src[:, 0], src[:, 1], s=9, color=C1, alpha=0.75, label="噪声 $x_1$（t=1）", zorder=3)
    ax1.scatter(tgt[:, 0], tgt[:, 1], s=9, color=C3, alpha=0.75, label="数据 $x_0$（t=0）", zorder=3)
    ax1.set_title("直线插值：$x_t=(1-t)x_0+t\\,x_1$", fontsize=12, color=INK)
    ax1.set_aspect("equal")
    ax1.legend(loc="lower right", frameon=False, fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])

    # ---- 右：Euler vs Midpoint 误差
    rng2 = np.random.default_rng(42)
    x_init = rng2.standard_normal(4000) * 0.5
    x_ref = _integrate(x_init.copy(), 1.0, 0.0, 20000, "midpoint")
    steps = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    e_err, m_err = [], []
    for n_steps in steps:
        e_err.append(np.mean((_integrate(x_init.copy(), 1.0, 0.0, n_steps, "euler") - x_ref) ** 2))
        m_err.append(np.mean((_integrate(x_init.copy(), 1.0, 0.0, n_steps, "midpoint") - x_ref) ** 2))
    ax2.loglog(steps, e_err, "o-", color=C0, lw=2, ms=4.5, label="Euler（1 NFE/步）")
    ax2.loglog(steps, m_err, "s-", color=C3, lw=2, ms=4.5, label="Midpoint（2 NFE/步）")
    ax2.set_xlabel("采样步数", fontsize=10)
    ax2.set_ylabel("与参考解的 MSE", fontsize=10)
    ax2.grid(color=GRID, which="both", lw=0.8)
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_title("曲线流场下：步数越多，误差越小", fontsize=12, color=INK)
    ax2.text(0.03, 0.06, "同 NFE 预算下，加密 Euler 步数往往优于高阶求解器",
             transform=ax2.transAxes, fontsize=8.8, color=MUTED)
    save(fig, "flow_trajectories.png")


# ---------------------------------------------------------------- 6. LoRA
def fig_lora() -> None:
    full = 4096 * 4096
    lora = 8 * 4096 + 4096 * 8
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars = ax.bar(["全量微调  W\n(4096×4096)", "LoRA  A+B\n(8×4096 ×2)"],
                  [full, lora], color=[C3, C0], width=0.52)
    ax.set_yscale("log")
    ax.set_ylabel("可训练参数量（log）", fontsize=10)
    ax.set_ylim(1e4, 1e8)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for b, v in zip(bars, [full, lora]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:,}", ha="center",
                fontsize=11, fontweight="bold", color=INK)
    ax.text(0.5, 1.02, f"可训练参数降至 {lora / full * 100:.2f}%", transform=ax.transAxes,
            ha="center", fontsize=11.5, color=C0)
    ax.tick_params(length=0, labelsize=9.5)
    save(fig, "lora_params.png")


# ---------------------------------------------------------------- 7. 3DGS 泼溅
def _gaussians_heart(n=2400, seed=11):
    """在隐式心形区域 (x²+y²−1)³ − x²y³ < 0 内均匀采样，模拟一组高斯。"""
    rng = np.random.default_rng(seed)
    chunks, total = [], 0
    while total < n:
        p = rng.uniform(-1.25, 1.25, (n, 2))
        x, y = p[:, 0], p[:, 1]
        inside = (x ** 2 + y ** 2 - 1.0) ** 3 - x ** 2 * y ** 3 < 0
        chunks.append(p[inside])
        total = sum(len(c) for c in chunks)
    pts = np.concatenate(chunks)[:n] * 0.72   # 留出边距，让心形轮廓完整可见
    xs, ys = pts[:, 0], pts[:, 1]

    scale = rng.uniform(0.012, 0.024, n)
    rot = rng.uniform(0, np.pi, n)
    depth = rng.uniform(0, 1, n)
    mix = np.clip((ys + 1.0) / 2.0, 0, 1)
    colors = np.stack([0.98 - 0.02 * mix, 0.14 + 0.46 * mix, 0.40 - 0.14 * mix], axis=1)
    opacity = rng.uniform(0.40, 0.78, n)
    return xs, ys, scale, rot, depth, colors, opacity


def _render_gaussians(W=640, H=570):
    xs, ys, scales, rots, depth, colors, opacity = _gaussians_heart()
    img = np.zeros((H, W, 3), dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    px = (xx / W - 0.5) * 2.0
    py = (0.5 - yy / H) * 2.0

    T = np.ones((H, W))
    for i in np.argsort(depth):  # 远 → 近
        c, s = np.cos(rots[i]), np.sin(rots[i])
        dx, dy = px - xs[i], py - ys[i]
        u = c * dx + s * dy
        v = -s * dx + c * dy
        # 各向异性椭圆：长轴 1.6 倍
        d2 = (u / (scales[i] * 1.7)) ** 2 + (v / (scales[i] * 0.9)) ** 2
        w = np.clip(np.exp(-0.5 * d2) * opacity[i], 0, 0.98)
        img += (T * w)[..., None] * colors[i][None, None, :]
        T *= (1 - w)

    img += T[..., None] * np.array([0.035, 0.045, 0.06])  # 深色背景
    return np.clip(img, 0, 1), (xs, ys, scales, rots, colors, opacity)


def fig_3dgs() -> None:
    img, params = _render_gaussians()
    xs, ys, scales, rots, colors, opacity = params

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                  gridspec_kw={"width_ratios": [1.02, 1]})
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("2400 个高斯经 Alpha 混合后的渲染结果", fontsize=12, color=INK, pad=8)

    ax2.set_xlim(-1.06, 1.06)
    ax2.set_ylim(-1.06, 1.06)
    ax2.set_aspect("equal")
    ax2.axis("off")
    rng = np.random.default_rng(2)
    idx = rng.choice(len(xs), 90, replace=False)
    for i in idx:
        c, s = np.cos(rots[i]), np.sin(rots[i])
        a, b = scales[i] * 1.7, scales[i] * 0.9
        th = np.linspace(0, 2 * np.pi, 90)
        ex, ey = a * np.cos(th), b * np.sin(th)
        ax2.fill(c * ex - s * ey + xs[i], s * ex + c * ey + ys[i],
                 facecolor=colors[i], alpha=0.10, edgecolor=colors[i], lw=0.5)
    ax2.scatter(xs[idx], ys[idx], s=4, color=colors[idx], alpha=0.95, zorder=3)
    ax2.set_title("每个高斯 = 位置 + 2D 协方差 + 颜色 + 不透明度", fontsize=11.5, color=INK, pad=8)
    ax2.text(0, -1.01, "按深度排序后从前到后累加，透光率 $T$ 递减", ha="center",
             fontsize=8.8, color=MUTED)
    save(fig, "3dgs_splat.png")


def main() -> None:
    print("生成插图 →", OUT)
    fig_roadmap()
    fig_cnn_pipeline()
    fig_vit_patches()
    fig_diffusion_forward()
    fig_flow()
    fig_lora()
    fig_3dgs()
    print("完成。")


if __name__ == "__main__":
    main()
