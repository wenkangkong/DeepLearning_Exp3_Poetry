import torch
import numpy as np
from model import PoetryLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的数据词汇表
data = np.load('data/poetry_data.npz', allow_pickle=True)
word2idx = data['word2idx'].item()
idx2word = data['idx2word'].item()
vocab_size = len(word2idx)

# 加载训练好的模型权重
model = PoetryLSTM(vocab_size).to(device)
model.load_state_dict(torch.load('models/poetry_model_final.pth', map_location=device))
model.eval()

def generate_poem(start_words, max_len=64, temperature=0.8):
    """生成古诗函数"""
    model.eval()
    # 将输入文字转为数字ID
    start_ids = [word2idx.get(ch, word2idx['<UNK>']) for ch in start_words]
    input_seq = torch.tensor([start_ids], dtype=torch.long).to(device)
    generated = start_words
    hidden = None
    
    for _ in range(max_len - len(start_ids)):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            # 取最后一个时间步的输出，并应用温度系数
            last_logits = output[0, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=0).cpu().numpy()
            # 根据概率随机采样下一个字
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = idx2word[next_idx]
            # 如果遇到结束标记，停止生成
            if next_char == '<END>':
                break
            generated += next_char
            # 将新生成的字符作为下一步的输入
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)
    return generated

if __name__ == '__main__':
    # 你可以在这里修改开头的几个字
    start = "水中月庭阁"
    print(f"输入开头: {start}")
    poem = generate_poem(start, temperature=0.8)
    print("生成的古诗:")
    print(poem)