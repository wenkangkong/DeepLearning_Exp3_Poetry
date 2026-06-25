import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from model import PoetryLSTM

# 1. 加载预处理数据
data = np.load('data/poetry_data.npz', allow_pickle=True)
sequences = data['data']
word2idx = data['word2idx'].item()
idx2word = data['idx2word'].item()
vocab_size = len(word2idx)

# 2. 定义数据集（滑动窗口）
class PoetryDataset(Dataset):
    def __init__(self, sequences, seq_len=64, max_poems=5000):  # 新增 max_poems 参数
        self.seq_len = seq_len
        self.samples = []
        # 只取前 max_poems 首诗，大幅减少计算量
        for seq in sequences[:max_poems]:  
            for i in range(0, len(seq) - seq_len - 1):
                self.samples.append((seq[i:i+seq_len], seq[i+1:i+seq_len+1]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# 3. 超参数设置
seq_len = 64
batch_size = 256  
epochs = 20        # 先跑20轮，大约需要1-2小时（CPU）或更快（GPU）
lr = 0.001

dataset = PoetryDataset(sequences, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PoetryLSTM(vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(f"使用设备: {device}")
print(f"训练样本总数: {len(dataset)}")
print("开始训练...")

# 4. 训练循环
for epoch in range(epochs):
    total_loss = 0
    loop = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1} 平均损失: {avg_loss:.4f}')
    
    # 每5轮保存一次模型
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'models/poetry_model_epoch{epoch+1}.pth')

# 5. 保存最终模型
torch.save(model.state_dict(), 'models/poetry_model_final.pth')
print("🎉 训练完成！模型已保存。")