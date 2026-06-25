import tkinter as tk
from tkinter import ttk, scrolledtext
import torch
import numpy as np
import os
import sys

# 导入同目录下的 model.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PoetryLSTM

# ---------- 1. 加载模型 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys

def resource_path(relative_path):
    """获取打包后资源的绝对路径"""
    try:
        # PyInstaller 会将临时文件解压到 _MEIPASS 目录
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# 使用 resource_path 获取数据文件路径
data_path = resource_path(os.path.join('data', 'poetry_data.npz'))
model_path = resource_path(os.path.join('models', 'poetry_model_final.pth'))
# 如果最终模型不存在，尝试备用的 epoch5
if not os.path.exists(model_path):
    model_path = resource_path(os.path.join('models', 'poetry_model_epoch5.pth'))
if not os.path.exists(model_path):
    model_path = os.path.join(base_dir, 'models', 'poetry_model_epoch5.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件，检查路径: {model_path}")

data = np.load(data_path, allow_pickle=True)
word2idx = data['word2idx'].item()
idx2word = data['idx2word'].item()
vocab_size = len(word2idx)

model = PoetryLSTM(vocab_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"模型加载成功，设备：{device}")

def generate_poem(start_words, max_len=64, temperature=0.8):
    model.eval()
    start_ids = [word2idx.get(ch, word2idx['<UNK>']) for ch in start_words]
    input_seq = torch.tensor([start_ids], dtype=torch.long).to(device)
    generated = start_words
    hidden = None
    for _ in range(max_len - len(start_ids)):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            last_logits = output[0, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=0).cpu().numpy()
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = idx2word[next_idx]
            if next_char == '<END>':
                break
            generated += next_char
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)
    return generated

# ----------  GUI  ----------
def on_generate():
    start = entry_start.get().strip()
    if not start:
        text_output.insert(tk.END, "⚠️ 请输入起始词！\n")
        return
    try:
        temp = float(slider_temp.get())
    except:
        temp = 0.8
    max_len = int(spinbox_len.get())
    poem = generate_poem(start, max_len=max_len, temperature=temp)
    text_output.insert(tk.END, f"🌸 起始词：{start}\n")
    text_output.insert(tk.END, f"🎛️ 温度：{temp:.2f}　|　📏 最大长度：{max_len}\n")
    text_output.insert(tk.END, f"📜 生成结果：\n{poem}\n")
    text_output.insert(tk.END, "─" * 50 + "\n")
    text_output.see(tk.END)

# 创建主窗口
root = tk.Tk()
root.title("🌸 智能古诗生成器")
root.geometry("680x580")
root.configure(bg="#f5f0eb")  # 柔和米灰背景

# 设置 ttk 主题
style = ttk.Style()
style.theme_use('clam')  # 'vista', 'xpnative' 等
style.configure('TFrame', background='#f5f0eb')
style.configure('TLabel', background='#f5f0eb', font=('微软雅黑', 11))
style.configure('TButton', font=('微软雅黑', 11), padding=6)
style.configure('TEntry', font=('微软雅黑', 12), padding=6)
style.configure('TScale', background='#f5f0eb')

# 主框架
main_frame = ttk.Frame(root, padding="15 15 15 15")
main_frame.pack(fill=tk.BOTH, expand=True)

# ---- 标题 ----
title_label = ttk.Label(main_frame, text="🌸 AI 古诗创作助手", font=('微软雅黑', 18, 'bold'), foreground="#5a4a3a")
title_label.pack(pady=(0, 10))

# 分隔线
ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=5)

# ---- 第一行：起始词（加大加粗） ----
row1 = ttk.Frame(main_frame)
row1.pack(fill=tk.X, pady=8)
ttk.Label(row1, text="📖 起始词：", font=('微软雅黑', 12)).pack(side=tk.LEFT, padx=(0, 10))
entry_start = ttk.Entry(row1, font=('微软雅黑', 14, 'bold'), width=30)  # 放大字体
entry_start.pack(side=tk.LEFT, fill=tk.X, expand=True)
entry_start.insert(0, "春眠不觉晓")

# ---- 第二行：温度滑块 ----
row2 = ttk.Frame(main_frame)
row2.pack(fill=tk.X, pady=8)
ttk.Label(row2, text="🌡️ 温度：", font=('微软雅黑', 11)).pack(side=tk.LEFT, padx=(0, 10))
slider_temp = tk.Scale(row2, from_=0.5, to=1.5, resolution=0.05,
                       orient=tk.HORIZONTAL, length=260,
                       bg='#f5f0eb', activebackground='#d9c8b2', troughcolor='#e0d6cc',
                       highlightthickness=0)
slider_temp.set(0.8)
slider_temp.pack(side=tk.LEFT, fill=tk.X, expand=True)
label_temp_val = ttk.Label(row2, text="0.80", font=('微软雅黑', 11))
label_temp_val.pack(side=tk.LEFT, padx=10)
def update_temp_label(val):
    label_temp_val.config(text=f"{float(val):.2f}")
slider_temp.config(command=update_temp_label)

# ---- 第三行：最大长度 ----
row3 = ttk.Frame(main_frame)
row3.pack(fill=tk.X, pady=8)
ttk.Label(row3, text="📏 最大长度：", font=('微软雅黑', 11)).pack(side=tk.LEFT, padx=(0, 10))
spinbox_len = tk.Spinbox(row3, from_=10, to=120, width=8, font=('微软雅黑', 11),
                         bg='#ffffff', relief='flat', borderwidth=1)
spinbox_len.delete(0, tk.END)
spinbox_len.insert(0, "64")
spinbox_len.pack(side=tk.LEFT)
ttk.Label(row3, text="(建议 40~80)", font=('微软雅黑', 10), foreground="#8a7a6a").pack(side=tk.LEFT, padx=10)

# ---- 生成按钮 ----
btn_generate = tk.Button(main_frame, text="✨ 生成古诗", command=on_generate,
                         bg="#b8a090", fg="white", font=('微软雅黑', 13, 'bold'),
                         relief='flat', padx=20, pady=8,
                         activebackground="#a08a7a", activeforeground="white",
                         cursor="hand2")
btn_generate.pack(pady=15)

# ---- 输出文本框 ----
text_output = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=14,
                                         font=('微软雅黑', 12),
                                         bg='#fcf8f4', fg='#3a2e26',
                                         relief='flat', borderwidth=2,
                                         highlightcolor='#b8a090',
                                         highlightthickness=1)
text_output.pack(fill=tk.BOTH, expand=True, pady=5)

# 启动
root.mainloop()