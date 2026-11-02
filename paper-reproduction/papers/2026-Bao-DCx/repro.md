# 复现记录 · DCx

> 状态：**环境受阻**（本机无 NVIDIA GPU）→ 已定位原因，待切换到云 GPU 继续。
> 原则：跑不通也要把"为什么跑不通、怎么修"写清楚，这比一个漂亮的数字更有信息量。

| 项目 | 内容 |
| :--- | :--- |
| 复现日期 | 2026.11.02 起 |
| 代码版本 | `DCx-main`（GitHub zip 下载解压，非 git clone，无 commit 记录） |
| 目标硬件 | 论文：4× RTX 4090；实际只需要 1 张 NVIDIA GPU |
| 本机 | Intel Iris Xe 核显（无 CUDA） |
| 当前结论 | ⚠️ 本机可读代码、可理解算法，**无法运行** |

---

## 1. 环境体检（2026.11.02 实测）

### 1.1 命令与结果

```powershell
> nvidia-smi
# 无输出（命令不存在）

> Get-CimInstance Win32_VideoController | Select Name
Intel(R) Iris(R) Xe Graphics        # 只有核显
```

```text
> conda env list
base      E:\anaconda
dcx       E:\anaconda\envs\dcx      # 已创建
learn_cnn E:\anaconda\envs\learn_cnn
```

```text
> E:\anaconda\envs\dcx\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.8.0+cpu False
```

| 检查项 | 期望（README） | 实际 | 结论 |
| :--- | :--- | :--- | :-: |
| Python | 3.9 | 3.9.25 | ✅ |
| PyTorch | 2.2.1 + cu118 | **2.8.0+cpu** | ❌ |
| CUDA | 11.8 | 无 | ❌ |
| `cubvh` | 已安装 | **未安装** | ❌ |
| `dcx_pkg` | `pip install .` 编译 | **未安装** | ❌ |
| 其他依赖 | open3d / trimesh / pyhocon / scikit-image / PyMCubes / point-cloud-utils | 均已安装 | ✅ |

已安装的其余依赖版本：`numpy 1.26.4`、`scipy 1.13.1`、`trimesh 4.12.2`、`open3d 0.19.0`、`pyhocon 0.3.57`、`scikit-image 0.24.0`、`PyMCubes 0.1.6`、`point-cloud-utils 0.29.7`。

### 1.2 报错原文

```text
ModuleNotFoundError: No module named 'cubvh'
ModuleNotFoundError: No module named 'dcx_pkg'
```

```text
TypeError: type torch.cuda.FloatTensor not available. Torch not compiled with CUDA enabled.
  File "...\torch\__init__.py", line 1264, in set_default_tensor_type
```

> 最后这个报错来自 `evaluate_finetune.py` 第 247 行：
> `torch.set_default_tensor_type('torch.cuda.FloatTensor')` —— 入口脚本**硬编码要求 CUDA**。

---

## 2. 诊断：为什么跑不起来

DCx 有三处**硬依赖** NVIDIA GPU / CUDA：

| # | 依赖点 | 后果 |
| :-: | :--- | :--- |
| 1 | `torch.set_default_tensor_type('torch.cuda.FloatTensor')` | 没有 CUDA 版 torch，脚本第二行就崩 |
| 2 | `cubvh`（GTUDF 模式） | 基于 CUDA 的 BVH，pip 安装时需编译，无 GPU 无法用 |
| 3 | `dcx_pkg`（C++ 扩展） | 虽然本身是 C++，但整个流水线传的是 CUDA 张量 |

**根因**：当前笔记本只有 Intel 核显，不存在"装个 CUDA 版 torch 就能跑"的可能。
`conda install pytorch-cuda=11.8` 在没有 NVIDIA 驱动的机器上会退化为 CPU 版本——这正是当前 `torch 2.8.0+cpu` 的来源。

---

## 3. 修复方案：换到云 GPU（推荐 AutoDL）

### 3.1 选镜像

- 实例：RTX 4090 / 3090（24 GB 显存足够 256³）
- 镜像：**PyTorch 2.2.1 + CUDA 11.8**（与 README 对齐，省去自己装）
- 数据盘：50 GB 起（2 亿采样点 + 中间结果很占空间）

### 3.2 环境步骤（照抄即可）

```bash
# 1. 确认 GPU 可用
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.2.1+cu118 True

# 2. 装 CUDA 工具链（编译 dcx_pkg 与 chamfer_dist 需要 nvcc）
conda install -c nvidia cuda-toolkit=11.8 cuda-nvcc=11.8 cuda-cccl=11.8 -y

# 3. Python 依赖
pip install open3d scikit-image tqdm pyhocon==0.3.57 trimesh PyMCubes scipy \
            point_cloud_utils==0.29.7 "numpy<2"

# 4. 编译 DCx 本体
cd DCX && pip install . && cd ..
python -c "import dcx_pkg; print('dcx ok')"

# 5. 编译 CAPUDF 的 Chamfer 扩展（NUDF 模式才需要）
cd CAPUDF/extensions/chamfer_dist && python setup.py install && cd ../../..

# 6. 安装 cubvh（GTUDF 模式需要）
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation
python -c "import cubvh; print('cubvh ok')"
```

### 3.3 自检清单（全部通过再往下走）

- [ ] `nvidia-smi` 有输出
- [ ] `torch.cuda.is_available() == True`
- [ ] `import dcx_pkg` 成功
- [ ] `import cubvh` 成功
- [ ] `nvcc --version` 显示 11.8

---

## 4. 数据准备

`evaluate_finetune.py` 的路径拼接规则：

```text
<conf: dir_path.data_dir>/<conf: dataset_dir>/<--datadir>/<--dataname>.ply
```

`test.conf` 里 `dataset_dir` 被注释掉了，所以实际是：

```text
./data/<--datadir>/<--dataname>.ply
```

### 4.1 先用一个玩具球验证链路

```python
# tools/make_toy_sphere.py（自己写，10 行）
import trimesh, os
os.makedirs("data/toy", exist_ok=True)
m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
m.export("data/toy/sphere.ply")
print("vertices:", len(m.vertices), "faces:", len(m.faces))
```

### 4.2 再换真实模型

论文的 45 个非流形模型没有公开下载；可以先用：
- Thingi10K / ABC 里自己挑的模型；
- 或 ShapeNet Car（论文报告了 108 个）。

> 注意：模型必须能被 `trimesh.load_mesh` 读入，且是三角网格。

---

## 5. 运行命令

### 5.1 冒烟测试（小分辨率 + 少采样点）

```bash
# 先把 confs/test.conf 里的 resolution 改成 64、iter_max 改成 100
python evaluate_finetune.py \
    --conf ./confs/test.conf \
    --datadir toy --dataname sphere \
    --udf_type 1 \
    --num_points 500000 \
    --gpu 0 --thinning --postprocessing
```

预期：`result/toy/sphere.ply` 生成，MeshLab 打开是一个球。

### 5.2 正式复现（GTUDF）

```bash
python evaluate_finetune.py \
    --conf ./confs/test.conf \
    --datadir <dataset> --dataname <shape> \
    --udf_type 1 \
    --num_points 200000000 \
    --gpu 0 --thinning --postprocessing --finetune
```

### 5.3 NUDF 模式

```bash
# 需要先把 CAPUDF 训练出的 ckpt 放到 ./ckpt/<dataset_dir>/<dataname>/ckpt_080000.pth
python evaluate_finetune.py \
    --conf ./confs/test.conf \
    --datadir <dataset> --dataname <shape> \
    --udf_type 0 \
    --gpu 0 --thinning --supplementary_sampling --postprocessing --finetune
```

---

## 6. 结果对照（待填）

| 场景 | 指标 | 论文 | 我的复现 | 差异 | 备注 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| GTUDF 球 | 是否闭合 | 是 | TODO | — | 冒烟测试 |
| GTUDF T 形件 | $CD_{glo}$ | — | TODO | — | 需要自建 GT |
| NUDF + CAPUDF | $CD_{glo}$ | 1.579e-3 | TODO | — | 论文 Table 5 |
| NUDF + CAPUDF | $CD_{nme}$ | 3.172e-3 | TODO | — | 论文 Table 5 |

**耗时对照（论文 Table 6，256³）**：UDF 查询 22 ms、投影 5984 ms、体素化 71 ms、细化 519 ms、补采样 1566 ms、V2M 107 ms、后处理 1588 ms。

---

## 7. 这次"跑不通"的收获

1. **"环境配好了"要定义清楚**：装完依赖 ≠ 能跑。判断标准是"入口脚本能跑到第一行业务逻辑"，而不是"pip install 没报错"。
2. **先看入口脚本的硬编码**：`torch.set_default_tensor_type('torch.cuda.FloatTensor')` 这种一行代码，比 README 更能说明硬件要求。
3. **CPU 版 torch 是静默降级**：没有 NVIDIA 显卡时，`conda install pytorch-cuda=11.8` 不会报错，只会给你一个 CPU 轮子。以后装完必须验证 `torch.cuda.is_available()`。
4. **复现的第一步是"拆分依赖"**：把流程拆成"必须 GPU"和"可以 CPU"两部分，就不会在一台没有 GPU 的机器上死磕。

---

## 8. 待办

- [ ] 申请 / 租用 AutoDL 实例（4090，PyTorch 2.2.1 + CUDA 11.8 镜像）
- [ ] 按 §3.2 配置，跑通 §3.3 自检清单
- [ ] 跑通 §5.1 冒烟测试，保存 `sphere.ply` 截图到 `assets/`
- [ ] 换 2–3 个真实模型，记录耗时与显存
- [ ] 对齐论文 Table 5 的 $CD_{glo}$ / $CD_{nme}$
- [ ] 做一次最小消融：去掉 `--thinning` 看非流形边变化
- [ ] 把结果填进 §6，更新状态为 ✅
