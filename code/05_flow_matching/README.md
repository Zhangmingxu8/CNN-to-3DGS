# 05 · Flow Matching 实践

对应笔记：[`docs/05-flow-matching.md`](../../docs/05-flow-matching.md)

## 文件

| 文件 | 作用 | 关键公式 |
| :--- | :--- | :--- |
| `flow_matching_toy.py` | 直线插值与速度场训练 | $x_t=(1-t)x_0+tx_1$，$v=x_1-x_0$ |
| `flow_matching_sampling_demo.py` | 10 步欧拉法采样 | $x \leftarrow x + v\cdot dt$ |
| `ode_solver_comparison_demo.py` | Euler vs Midpoint 误差对比 | 大步长下差一个数量级 |

## 运行

```bash
python code/05_flow_matching/flow_matching_toy.py
python code/05_flow_matching/flow_matching_sampling_demo.py
python code/05_flow_matching/ode_solver_comparison_demo.py
```

## 预期输出

```text
大步长 (dt = -0.5) 下的单步截断误差对比:
一阶欧拉法 (Euler) MSE 误差:   0.0xxxxx
二阶中点法 (Midpoint) MSE 误差: 0.0xxxxx
中点法精度提升倍数:            x.xx 倍
```

## 注意

采样脚本中的速度网络同样是未训练的，输出不代表生成质量，只用于观察积分行为与 `dt` 方向。
