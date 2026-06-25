import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 解决matplotlib中文显示问题（Windows 通常用 SimHei 或 Microsoft YaHei）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置项目根目录（因为脚本在 src/ 下，需要回到上一级）
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_raw_dir = os.path.join(base_dir, '..', 'chinese-poetry')  # 根据你的实际路径调整
output_dir = os.path.join(base_dir, 'runs')  # 图表保存位置

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 加载原始数据（复用 preprocess 的逻辑）
def load_all_poems(base_dir):
    poems = []
    print(f"正在扫描目录: {base_dir}")
    if not os.path.exists(base_dir):
        print(f"⚠️ 警告: 路径 {base_dir} 不存在，请修改 eda.py 中的 data_raw_dir 变量！")
        return []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
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
                except:
                    continue
    return poems

print("📚 正在加载古诗数据...")
# 注意：如果你的 chinese-poetry 文件夹就在 poetry_generation 同级，这里路径可能需要调整
# 这里默认往回退两层去找 chinese-poetry，如果不对请改成你的绝对路径，例如 r'E:\pythontask\chinese-poetry'
poems = load_all_poems(os.path.join(base_dir, '..', 'chinese-poetry'))  

if not poems:
    print("❌ 没有找到任何诗，请检查 eda.py 第 10 行附近的 data_raw_dir 路径！")
    sys.exit()

print(f"✅ 成功加载 {len(poems)} 首诗")

# 2. 计算统计量
lengths = [len(p) for p in poems]
all_chars = ''.join(poems)
char_counter = Counter(all_chars)
total_chars = len(all_chars)
vocab_size = len(char_counter)

print("\n" + "="*40)
print("📊 数据集统计报告")
print("="*40)
print(f"总诗数: {len(poems)}")
print(f"总字符数（含重复）: {total_chars}")
print(f"去重后的字符数（词汇量）: {vocab_size}")
print(f"平均诗长: {np.mean(lengths):.2f} 字符")
print(f"最长诗: {max(lengths)} 字符")
print(f"最短诗: {min(lengths)} 字符")
print(f"最常见的字: '{char_counter.most_common(1)[0][0]}' (出现 {char_counter.most_common(1)[0][1]} 次)")
print("="*40 + "\n")

# 3. 绘图1：诗歌长度分布（柱状图 + 密度曲线）
fig1, ax1 = plt.subplots(figsize=(10, 6))
# 分桶，因为古诗大多是 20~60 字，桶距设为 2 或 4
bins = range(min(lengths), max(lengths) + 4, 2)  
ax1.hist(lengths, bins=bins, edgecolor='black', alpha=0.7, color='#5B8FF9', label='诗数量')
ax1.set_xlabel('诗歌长度（字符数）', fontsize=12)
ax1.set_ylabel('诗篇数', fontsize=12)
ax1.set_title('全唐诗长度分布（EDA）', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
# 标注均值和峰值
mean_len = np.mean(lengths)
ax1.axvline(mean_len, color='red', linestyle='--', linewidth=2, label=f'平均长度: {mean_len:.1f}')
ax1.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'poem_length_distribution.png'), dpi=300)
print(f"✅ 图表1已保存: {os.path.join(output_dir, 'poem_length_distribution.png')}")

# 4. 绘图2：高频字统计（前20个）
fig2, ax2 = plt.subplots(figsize=(12, 6))
top_words = char_counter.most_common(20)
words, counts = zip(*top_words)
# 使用渐变色条形图
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))[::-1]
bars = ax2.bar(words, counts, color=colors, edgecolor='black')
ax2.set_xlabel('汉字', fontsize=12)
ax2.set_ylabel('出现频次', fontsize=12)
ax2.set_title('全唐诗高频字 TOP 20', fontsize=14, fontweight='bold')
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
# 在柱子上标数字
for bar, count in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
             str(count), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_chars_frequency.png'), dpi=300)
print(f"✅ 图表2已保存: {os.path.join(output_dir, 'top_chars_frequency.png')}")

# 打印前5首诗作为样本
print("\n📜 样本示例（前5首诗）:")
for i, p in enumerate(poems[:5]):
    print(f"--- 诗 {i+1} ---")
    # 可以加上简单的排版，如每5个字加空格
    formatted = ' '.join(p[j:j+5] for j in range(0, len(p), 5)) if len(p) > 5 else p
    print(formatted)
    
print(f"\n🎉 EDA 完成！所有图表已保存在: {output_dir}")