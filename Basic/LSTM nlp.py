import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 示例文本数据
text = "We are learning to build a language model with LSTM networks in NLP"

# 预处理文本：分词和创建词汇表
words = text.split()
vocab = set(words)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)

# 超参数
embedding_dim = 10
hidden_dim = 20
learning_rate = 0.01
num_epochs = 200

# 构建数据集，生成 (input, target) 对
sequence_length = 3  # 输入序列的长度
data = []
for i in range(len(words) - sequence_length):
    input_seq = words[i:i + sequence_length]
    target_word = words[i + sequence_length]
    data.append((input_seq, target_word))

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_word = self.data[idx]
        input_ids = torch.tensor([word_to_idx[word] for word in input_seq], dtype=torch.long)
        target_id = torch.tensor(word_to_idx[target_word], dtype=torch.long)
        return input_ids, target_id

dataset = TextDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
        return output

# 初始化模型、损失函数和优化器
model = LSTMModel(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试模型预测下一词
def predict_next_word(model, input_text):
    model.eval()
    words = input_text.split()
    input_ids = torch.tensor([word_to_idx[word] for word in words], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids)
        predicted_id = torch.argmax(output, dim=1).item()
    return idx_to_word[predicted_id]

# 示例预测
input_text = "language model with"
predicted_word = predict_next_word(model, input_text)
print(f"Input text: '{input_text}' -> Predicted next word: '{predicted_word}'")
