"""
1、
【模块名称】LLM 核心前向流与 Tokenizer 语义映射 (Token Flow Demo)
【解决痛点】打通大语言模型（LLM）从纯文字输入到整数 Token、词嵌入向量、因果自注意力到预测下一个 Token 的完整流水线。
【核心流程】
  1. 模拟 Tokenizer 分词与字典查找 (文字 -> 整数 ID)
  2. nn.Embedding 查表映射为高维张量 [Batch, Seq_Len, Embed_Dim]
  3. 叠加因果自注意力 (Causal Attention) 只能看前文
  4. 线性头 (LM Head) 映射回词表大小，预测下一个最可能的 Token ID
【张量流动】文字 -> 整数序列 [1, 4] -> 嵌入 [1, 4, 32] -> 掩码注意力 [1, 4, 32] -> 输出词表得分 [1, 4, 1000]
"""

import torch
import torch.nn as nn
import math

# 1. 模拟环境参数
vocab_size = 1000   # 假设字典里收录了 1000 个常见词/词元
embed_dim = 32      # 每个词映射为 32 维特征
seq_len = 4         # 输入句子包含 4 个 Token

# 2. 模拟真实文本转换为 Token ID
# 假设输入一句话："我 爱 深度 学习"，通过分词器查表得到了 4 个整数 ID
input_tokens = torch.tensor([[101, 234, 567, 890]])  # 形状: [Batch=1, Seq_Len=4]
print("1. 输入文本被 Tokenizer 转换后的整数序列 ID:", input_tokens)

# 3. 词嵌入层 (Embedding Layer)
# 它是一张 [1000, 32] 的大表格，负责把整数 ID 换成 32 维连续向量
token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
x = token_embedding(input_tokens)  # 形状: [1, 4, 32]
print("2. 查表后的词嵌入向量形状 [Batch, Seq_Len, Embed_Dim]:", x.shape)

# 4. 因果自注意力层 (只能看前文，绝不偷看未来)
w_q = nn.Linear(embed_dim, embed_dim, bias=False)
w_k = nn.Linear(embed_dim, embed_dim, bias=False)
w_v = nn.Linear(embed_dim, embed_dim, bias=False)

Q = w_q(x)
K = w_k(x)
V = w_v(x)

# 计算原始打分
scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(embed_dim)
# 构建严格的上三角掩码，遮蔽未来
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
masked_scores = scores.masked_fill(mask, float('-inf'))
attn_weights = torch.softmax(masked_scores, dim=-1)

# 加权聚合
context = torch.matmul(attn_weights, V)  # 形状: [1, 4, 32]
print("3. 经过因果自注意力聚合后的特征形状:", context.shape)

# 5. 语言模型分类头 (LM Head)
# 把 32 维向量映射回词表大小 1000，为词表里的 1000 个候选词挨个打分！
lm_head = nn.Linear(embed_dim, vocab_size)
logits = lm_head(context)  # 形状: [1, 4, 1000]
print("4. 最终在整个词表上的预测打分形状 [Batch, Seq_Len, Vocab_Size]:", logits.shape)

# 6. 预测下一个词 (Next Token Prediction)
# 我们只关心序列最后一个词 (位置 -1) 吐出来的预测得分
last_token_logits = logits[0, -1, :]  # 长度 1000 的向量
predicted_next_token_id = torch.argmax(last_token_logits).item()
print(f"\n模型预测的下一个紧随其后的 Token ID 是: {predicted_next_token_id}")