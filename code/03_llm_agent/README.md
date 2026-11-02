# 03 · LLM & Agent 实践

对应笔记：[`docs/03-llm-agent.md`](../../docs/03-llm-agent.md)

## 文件

| 文件 | 作用 | 观察点 |
| :--- | :--- | :--- |
| `token_flow_demo.py` | 文本 → Token ID → Embedding → 因果注意力 → 词表打分 | `[1,4] → [1,4,32] → [1,4,1000]` |
| `autoregressive_loop_demo.py` | 自回归生成循环，逐词拼接 | 每步只取最后一位的 argmax |
| `lora_toy_demo.py` | LoRA 低秩旁路与参数量统计 | 可训练参数 ≈ 0.39% |
| `simple_agent_demo.py` | Function Calling 的 JSON 协议与执行循环 | 模型只决策，外部函数干实际工作 |

## 运行

```bash
python code/03_llm_agent/token_flow_demo.py
python code/03_llm_agent/autoregressive_loop_demo.py
python code/03_llm_agent/lora_toy_demo.py
python code/03_llm_agent/simple_agent_demo.py
```

## 预期输出

```text
1. 输入文本被 Tokenizer 转换后的整数序列 ID: tensor([[101, 234, 567, 890]])
2. 查表后的词嵌入向量形状 [Batch, Seq_Len, Embed_Dim]: torch.Size([1, 4, 32])
...
模型预测的下一个紧随其后的 Token ID 是: xxx
```

```text
被冻结的原厂主干参数量: 16,777,216
实际需要训练的 LoRA 参数量: 65,536
可训练参数比例仅为: 0.39%
```

## 注意

- 这里的模型都是**随机初始化**的，输出没有语义，只用于验证数据流；
- `simple_agent_demo.py` 中的 `eval()` 仅为教学演示，生产环境必须替换为安全解析器。
