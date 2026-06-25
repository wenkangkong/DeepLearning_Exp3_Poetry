# 深度学习实验 Exp3：中文古诗生成（基于 LSTM）

> 本项目是深度学习课程的实验三，利用双层 LSTM 构建字符级语言模型，自动生成中文古诗。

## 功能特点

- 基于 PyTorch 的双层 LSTM 模型，支持 GPU 加速训练
- 字符级词汇表构建，无需分词
- 交互式图形界面（Tkinter），支持用户输入起始词并调节温度参数
- 自动生成符合古诗风格的完整诗句
- 完整的 EDA 可视化和训练损失曲线

## 项目结构
.
├── src/ # 源代码目录
│ ├── preprocess.py # 数据预处理
│ ├── eda.py # 探索性数据分析（生成图表）
│ ├── model.py # LSTM 模型定义
│ ├── train.py # 训练脚本
│ ├── generate.py # 诗歌生成脚本（命令行）
│ ├── gui.py # 图形界面（Tkinter）
│ ├── eval_loss.py # 重新计算训练损失
│ └── plot_loss_from_data.py # 绘制损失曲线图
├── data/ # 数据目录（预处理后的文件）
├── models/ # 模型检查点（训练后生成）
├── runs/ # 训练日志和图表输出
└── README.md

text

## ⚙️ 环境配置

### 硬件要求
- CPU：8 核以上（推荐）
- GPU：NVIDIA GTX 1060+（显存 ≥6GB，可选）

### 软件依赖
- Python 3.8 ~ 3.12
- PyTorch 2.5.1+（GPU 版本需 CUDA 12.6 兼容）
- 其他库见 `requirements.txt`

### 安装步骤（使用 venv）
```bash
# 克隆本仓库
git clone https://github.com/wenkangkong/DeepLearning_Exp3_Poetry.git
cd DeepLearning_Exp3_Poetry

# 创建虚拟环境并激活（Windows）
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
