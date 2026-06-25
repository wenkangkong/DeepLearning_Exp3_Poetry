import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model import PoetryLSTM
import os

# 加载数据
data = np.load('data/poetry_data.npz', allow_pickle=True)
sequences = data['data']
word2idx = data['word2idx'].item()
vocab_size = len(word2idx)

# 与 train.py 相同的 Dataset 定义（滑动窗口）
class PoetryDataset(Dataset):
    def __init__(self, sequences, seq_len=64):
        self.seq_len = seq_len
        self.samples = []
        for seq in sequences[:5000]:  # 与训练时一致，取前5000首
            for i in range(0, len(seq) - seq_len - 1):
                self.samples.append((seq[i:i+seq_len], seq[i+1:i+seq_len+1]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

seq_len = 64
batch_size = 256  # 与训练一致
dataset = PoetryDataset(sequences, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # shuffle=False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])

# 加载你保存的模型文件（可能多个）
checkpoints = [
    'models/poetry_model_epoch5.pth',
    'models/poetry_model_epoch10.pth',
    'models/poetry_model_epoch15.pth',
    'models/poetry_model_final.pth'
]

losses = []
for ckpt_path in checkpoints:
    if not os.path.exists(ckpt_path):
        print(f"⚠️ {ckpt_path} 不存在，跳过")
        continue
    model = PoetryLSTM(vocab_size).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    total_loss = 0
    total_batches = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item() * x.size(0)  # 按样本数加权
            total_batches += x.size(0)
    avg_loss = total_loss / total_batches
    losses.append((ckpt_path, avg_loss))
    print(f"{ckpt_path} 平均损失: {avg_loss:.4f}")

# 输出结果，你可以直接抄到报告里
print("\n✅ 计算完成！可用于绘图的损失列表（按顺序从第5轮到最终轮）：")
for name, loss in losses:
    print(f"{loss:.4f}")