# 机器学习底层推导

用 NumPy 把反向传播和 PCA 推到能跑的代码。本仓库不再更新。

大模型训练与推理精读见 [nanochat-guide](https://github.com/smiletrl/nanochat-guide)。

## 内容

| # | 项目 | 说明 |
|---|---|---|
| 01 | [MNIST：从零手写神经网络](./01_nn_from_scratch_mnist) | NumPy 手推反向传播，对照可运行训练脚本 |
| 02 | [PCA：从代数推导到 SVD](./02_pca_math_to_svd) | 协方差特征值分解与 SVD 的等价性，对照 sklearn |

推导从 [MNIST 公式文档](01_nn_from_scratch_mnist/docs/README.md) 开始。跑通训练见 [MNIST 快速开始](01_nn_from_scratch_mnist/README.md#快速开始-quick-start)。

## 环境

使用 [uv](https://github.com/astral-sh/uv)：

```bash
git clone git@github.com:smiletrl/machine-learning.git
cd machine-learning
uv sync
source .venv/bin/activate
```


