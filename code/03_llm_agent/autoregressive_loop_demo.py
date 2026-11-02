"""
2、
【模块名称】大语言模型自回归生成循环 (Auto-regressive Generation Loop)
【解决痛点】亲眼看清大模型是怎么一个词一个词往外蹦，并重新拼回输入端的底层机制。
【核心流程】
  1. 输入初始 Prompt 的 Token 序列
  2. 经过 Embedding + 因果注意力 + LM Head 计算
  3. 抓取最后一个位置的预测结果，选出概率最大的下一个 Token ID (Greedy Search)
  4. 使用 torch.cat 将新 Token 拼到输入末尾，重复循环
  5. 遇到终止符 <EOS> (ID=0) 立即提前结束生成
【张量流动】输入: [1, 3] -> 预测并拼接 -> [1, 4] -> 预测并拼接 -> [1, 5] ...
"""

import torch
import torch.nn as nn
import math

# 1. 基础配置
vocab_size = 50  # 假设词表里只有 50 个词
embed_dim = 16  # 隐藏维度 16
max_new_tokens = 5  # 最多让它往外蹦 5 个新词

# 构造一个极简字典用于模拟展示
vocab = {i: f"词语_{i}" for i in range(vocab_size)}
vocab[0] = "<EOS>"  # 0 号 Token 约定为句子结束符


# 2. 组装一个极简 LLM (Embedding + 因果注意力 + 分类头)
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        x = self.embedding(input_ids)

        # 产生 Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 因果掩码注意力 (只能看过去，不能偷看未来)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(embed_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # 映射回整个词表打分: [Batch, Seq_Len, Vocab_Size]
        logits = self.lm_head(context)
        return logits


# 3. 初始化模型并进入评估推理模式
model = MiniLLM(vocab_size, embed_dim)
model.eval()

# 模拟用户输入的开场白 Prompt (包含 3 个 Token)
current_input = torch.tensor([[12, 5, 23]])
print(f"用户输入的 Prompt Token 序列: {current_input[0].tolist()}")

# 4. 核心：自回归循环 (一步步吐字)
print("\n--- 大模型开始逐字生成 ---")
with torch.no_grad():
    for step in range(max_new_tokens):
        # 步骤 A: 喂入当前手里的所有 Token，模型前向推理
        logits = model(current_input)

        # 步骤 B: 核心关键！我们永远只拿【最后一个位置】预测出来的得分！
        last_token_logits = logits[:, -1, :]  # 形状: [1, 50]

        # 步骤 C: 挑选得分最高的那个词作为下一个词 (贪心搜索 Greedy Search)
        next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)  # 形状: [1, 1]

        # 步骤 D: 拼接！把新预测出来的词追加到末尾 (这就是所谓的 Autoregressive！)
        current_input = torch.cat([current_input, next_token], dim=1)

        print(f"第 {step + 1} 步: 吐出新 Token ID: {next_token.item()} | 当前累计长句: {current_input[0].tolist()}")

        # 步骤 E: 遇到结束符提前终止
        if next_token.item() == 0:
            print("模型认为话已说完，吐出了 <EOS> 终止符，停止生成！")
            break

print(f"\n最终完整生成的序列: {current_input[0].tolist()}")