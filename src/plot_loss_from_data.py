import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import os

# 设置中文字体（Windows 用微软雅黑，Mac 用 Arial Unicode MS）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 你刚刚得到的数据 ----------
epochs = [5, 10, 15, 20]                    # 轮次
losses = [1.1791, 0.4342, 0.3056, 0.2600]  # 对应损失

# ---------- 插值生成平滑曲线（让曲线更美观） ----------
epochs_dense = np.linspace(5, 20, 300)  # 在5~20之间生成300个点
spline = make_interp_spline(epochs, losses, k=3)  # 三次样条插值
losses_smooth = spline(epochs_dense)

# ---------- 绘图（论文风格） ----------
fig, ax = plt.subplots(figsize=(10, 6))

# 平滑曲线
ax.plot(epochs_dense, losses_smooth, linewidth=2.5, color='#2E86AB', label='训练损失（插值平滑）')

# 原始数据点（用大圆点标注）
ax.scatter(epochs, losses, s=120, color='#D94F4F', zorder=5, label='实际测量点', edgecolors='black', linewidth=1.2)

# 在每个数据点旁标注数值
for x, y in zip(epochs, losses):
    ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, 12), 
                ha='center', fontsize=10, fontweight='bold', color='#D94F4F')

# 坐标轴与标题
ax.set_xlabel('训练轮次 (Epoch)', fontsize=13, fontweight='bold')
ax.set_ylabel('交叉熵损失 (Cross-Entropy Loss)', fontsize=13, fontweight='bold')
ax.set_title('中文古诗生成模型 - 训练损失曲线', fontsize=15, fontweight='bold', pad=15)

# 网格与图例
ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
ax.legend(loc='upper right', fontsize=11)

# 设置x轴刻度为整数
ax.set_xticks(np.arange(4, 21, 2))

# 去掉顶部和右侧边框线（更清爽）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------- 保存高清图片 ----------
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_path = os.path.join(output_dir, 'training_loss_curve_professional.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 损失曲线已保存至: {save_path}")

# 显示图片
plt.show()

# ---------- 同时打印一个简洁的表格，方便你写进报告 ----------
print("\n" + "="*50)
print("📊 训练损失统计表（可直接复制到报告）")
print("="*50)
print("| 轮次 | 平均损失 |")
print("|------|----------|")
for e, l in zip(epochs, losses):
    print(f"| {e:4d}  | {l:.4f}   |")
print("="*50)
print(f"📉 总下降幅度: {losses[0] - losses[-1]:.4f}")
print(f"📉 下降百分比: {(1 - losses[-1]/losses[0])*100:.1f}%")