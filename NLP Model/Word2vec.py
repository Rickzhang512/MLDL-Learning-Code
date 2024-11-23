import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

# 数据预处理
text = "We are learning natural language processing with neural networks"
words = text.lower().split()
vocab = set(words)
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}


# 准备Skip-gram数据
def generate_skipgram_pairs(words, window_size=2):
    pairs = []
    for i, word in enumerate(words):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                pairs.append((word_to_idx[word], word_to_idx[words[j]]))
    return pairs


skipgram_pairs = generate_skipgram_pairs(words)


# 定义 Word2Vec 模型
class Word2VecNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embed_size, num_negatives=5):
        super(Word2VecNegativeSampling, self).__init__()
        self.embed_v = nn.Embedding(vocab_size, embed_size)
        self.embed_u = nn.Embedding(vocab_size, embed_size)
        self.num_negatives = num_negatives

    def forward(self, center_word, context_word, negative_samples):
        # 中心词和上下文词的嵌入向量
        v = self.embed_v(center_word)
        u = self.embed_u(context_word)

        # 计算正样本的分数并取 sigmoid
        pos_score = torch.mul(v, u).sum(dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        # 负样本的分数
        neg_score = torch.bmm(negative_samples.unsqueeze(1), self.embed_u(context_word).unsqueeze(2)).squeeze()
        neg_loss = torch.log(torch.sigmoid(-neg_score)).sum(dim=1)

        # 总损失：正样本损失和负样本损失
        return -(pos_loss + neg_loss).mean()


# 负采样函数
def get_negative_samples(context_word, num_negatives):
    """随机采样负样本"""
    negative_samples = []
    while len(negative_samples) < num_negatives:
        negative_word = np.random.choice(list(range(vocab_size)))
        if negative_word != context_word:
            negative_samples.append(negative_word)
    return torch.LongTensor(negative_samples)


# 训练模型
embed_size = 10
model = Word2VecNegativeSampling(vocab_size, embed_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for center, context in skipgram_pairs:
        center = torch.LongTensor([center])
        context = torch.LongTensor([context])
        negative_samples = get_negative_samples(context.item(), model.num_negatives)

        optimizer.zero_grad()
        loss = model(center, context, negative_samples)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(skipgram_pairs)}")
