import json
import os
import numpy as np
from collections import Counter

def load_all_poems(base_dir):
    """自动遍历所有子文件夹，加载所有JSON文件中的古诗"""
    poems = []
    print(f"正在扫描目录: {base_dir}")
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        # 处理不同的JSON结构
                        if isinstance(content, list):
                            for item in content:
                                if 'paragraphs' in item:
                                    poem_text = ''.join(item['paragraphs'])
                                    if 20 < len(poem_text) < 128:
                                        poems.append(poem_text)
                        elif isinstance(content, dict) and 'paragraphs' in content:
                            poem_text = ''.join(content['paragraphs'])
                            if 20 < len(poem_text) < 128:
                                poems.append(poem_text)
                except Exception as e:
                    # 忽略无法解析的文件
                    continue
    return poems

def build_vocab(poems):
    all_chars = ''.join(poems)
    counter = Counter(all_chars)
    chars = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)
    chars = ['<PAD>', '<UNK>', '<START>', '<END>'] + chars
    word2idx = {ch: idx for idx, ch in enumerate(chars)}
    idx2word = {idx: ch for idx, ch in enumerate(chars)}
    return word2idx, idx2word

def preprocess_poems(poems, word2idx, max_len=125):
    sequences = []
    for poem in poems:
        seq = [word2idx['<START>']] + [word2idx.get(ch, word2idx['<UNK>']) for ch in poem] + [word2idx['<END>']]
        if len(seq) <= max_len:
            seq = seq + [word2idx['<PAD>']] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        sequences.append(seq)
    return np.array(sequences, dtype=np.int64)

if __name__ == '__main__':
    # ！！！！请修改这里为你的实际路径！！！！
    base_dir = r'E:\pythontask\chinese-poetry'  
    
    print("开始加载古诗数据...")
    poems = load_all_poems(base_dir)
    print(f"成功加载 {len(poems)} 首诗")
    
    if len(poems) == 0:
        print("错误：没有找到任何古诗，请检查 base_dir 路径是否正确！")
        exit()
    
    word2idx, idx2word = build_vocab(poems)
    print(f"词汇表大小（字符数）: {len(word2idx)}")
    
    sequences = preprocess_poems(poems, word2idx)
    print(f"生成的数据形状: {sequences.shape}")
    
    # 保存到 data 文件夹
    np.savez('data/poetry_data.npz', data=sequences, word2idx=word2idx, idx2word=idx2word)
    print("✅ 预处理完成！文件已保存到 data/poetry_data.npz")