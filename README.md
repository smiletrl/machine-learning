# 🚀 机器学习底层推导与工程实战

## 📖 关于这个仓库

机器学习项目练习。大模型LLM训练，参考另一个repository [LLM](https://github.com/smiletrl/llm)

*快速体验：你可以通过 [公式推导手册](01_nn_from_scratch_mnist/docs/README.md) 理解神经网络底层的反向传播，并配合 [快速开始指南](01_nn_from_scratch_mnist/README.md#快速开始-quick-start) 亲手跑通纯 NumPy 的神经网络。5 个 Epoch 训练总耗时仅需 0.6 秒，准确率直飙 96%+！*

## 🛠️ 本地开发环境 (Local Setup)

本项目拥抱现代化的 Python 工具链，使用极致快速的 [uv](https://github.com/astral-sh/uv) 作为包和项目管理工具。

```bash
# 1. 克隆仓库并进入目录
git clone git@github.com:smiletrl/machine_learning.git
cd machine_learning

# 2. 一键同步并安装全部依赖 (uv 会自动为你创建虚拟环境 .venv)
uv sync

# 3. 激活当前项目的虚拟环境
source .venv/bin/activate
```
## 🗂️ 项目索引 (Projects)

📺 **完整逐层公式推导讲解视频，请关注小红书 & 抖音平台，账号：@清影Labs。**

| 编号 | 项目名称 | 核心技术点 | 状态 | 
| ----- | ----- | ----- | ----- | 
| 01 | [从零手撕神经网络 (MNIST 手写数字识别)](./01_nn_from_scratch_mnist) | MNIST 识别实战, NumPy 手推反向传播 | 🟢 已完成 | 
| 02 | [主成分分析 (PCA)：从纯代数推导到 SVD 工业级实现](./02_pca_math_to_svd) | SVD 等价性证明, 数值稳定性, Numpy 白盒复现 | 🟢 已完成 | 
| 03 | [短视频推荐系统：从底层推导到工业落地](./03_short_video_recommendation) | 纯 NumPy 手撕 Attention, 双塔召回, 负采样, 序列建模 | 🔥 研发中 | 
| 04 | [经典机器学习算法核心破壁 (持续更新)](#) | 决策树, SVM, K-Means 核心逻辑手推 | 🟡 规划中 | 
| 05 | [凸优化理论与代码实战 (规划中)](#) | 梯度下降, 拉格朗日对偶, 损失曲面 | 🟡 规划中 | 

---
*Follow my journey bridging high-performance backend engineering with hardcore AI computation. Give it a ⭐️ if it inspires you!*