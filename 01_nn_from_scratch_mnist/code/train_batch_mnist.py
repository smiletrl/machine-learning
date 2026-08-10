import numpy as np
from datasets import load_dataset
import time

# ==========================================
# 1. 加载并打包数据 (引入测试集)
# ==========================================
print("正在加载数据集...")
ds = load_dataset("ylecun/mnist")

# --- 准备训练集 (日常作业，60000张) ---
X_train = np.array([np.array(img).flatten() for img in ds['train']['image']]).T # (784, 60000)
X_train = X_train.astype('float32') / 255.0

labels_train = np.array(ds['train']['label'])
Y_train = np.zeros((10, len(labels_train)))
for i, label in enumerate(labels_train):
    Y_train[label, i] = 1.0                                                    # (10, 60000)

# --- 准备测试集 (期末考试，10000张，完全不参与训练) ---
print("正在准备期末考试卷(测试集)...")
X_test = np.array([np.array(img).flatten() for img in ds['test']['image']]).T   # (784, 10000)
X_test = X_test.astype('float32') / 255.0

# 测试集只需要核对对错，所以不需要转成 One-Hot，保留原始数字标签即可
labels_test = np.array(ds['test']['label'])                                     # (10000,)


# ==========================================
# 2. 神经网络组件
# ==========================================
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

# 初始化权重 (784 -> 16 -> 15 -> 10)
W0 = np.random.randn(16, 784) * 0.01  
b0 = np.zeros((16, 1))

W1 = np.random.randn(15, 16) * 0.01
b1 = np.zeros((15, 1))

W2 = np.random.randn(10, 15) * 0.01
b2 = np.zeros((10, 1))

audit = 0

# ==========================================
# 3. 核心算法 (前向、反向、以及纯前向的考试机制)
# ==========================================
def train_batch_step(X_batch, Y_batch, learning_rate):
    """训练机制：看题 -> 对答案 -> 纠正脑回路 (权重)"""
    global W0, b0, W1, b1, W2, b2, audit
    batch_size = X_batch.shape[1]

    
    # 前向传播
    h1 = np.dot(W0, X_batch) + b0
    a1 = relu(h1)
    h2 = np.dot(W1, a1) + b1
    a2 = relu(h2)
    h3 = np.dot(W2, a2) + b2
    p = softmax(h3)

    if audit == 0:
        print(f"W0 shape: {W0.shape}, X_batch shape: {X_batch.shape}, b0 shape: {b0.shape},p shape: {p.shape}, y shape: {Y_batch.shape}")
        rulin = audit
    
    # 反向传播
    g3 = p - Y_batch
    dW2 = np.dot(g3, a2.T) / batch_size   
    db2 = np.sum(g3, axis=1, keepdims=True) / batch_size
    
    g2 = np.dot(W2.T, g3) * relu_deriv(h2)
    dW1 = np.dot(g2, a1.T) / batch_size   
    db1 = np.sum(g2, axis=1, keepdims=True) / batch_size
    
    g1 = np.dot(W1.T, g2) * relu_deriv(h1)
    dW0 = np.dot(g1, X_batch.T) / batch_size
    db0 = np.sum(g1, axis=1, keepdims=True) / batch_size
    
    # 参数更新
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W0 -= learning_rate * dW0
    b0 -= learning_rate * db0

def evaluate(X, true_labels):
    """考试机制：只做题不看答案，不更新权重。测试真正的泛化能力。"""
    # 纯前向传播，让 10000 张图瞬间穿过网络
    h1 = np.dot(W0, X) + b0
    a1 = relu(h1)
    h2 = np.dot(W1, a1) + b1
    a2 = relu(h2)
    h3 = np.dot(W2, a2) + b2
    p = softmax(h3)
    
    # 选出概率最大的那个数字作为预测结果
    predictions = np.argmax(p, axis=0)
    
    # 计算准确率
    accuracy = np.mean(predictions == true_labels) * 100
    return accuracy

# ==========================================
# 4. 训练与测试循环
# ==========================================
epochs = 5
batch_size = 64
learning_rate = 0.5 # 初始步子很大
lr_decay = 0.8      # 【新增】衰减系数：每跑完一轮，步子缩小到原来的 80%
num_samples = X_train.shape[1]

print(f"\n=== 开始硬核实战 (Batch Size: {batch_size}) ===")
for epoch in range(epochs):
    start_time = time.time()
    
    # --- 阶段 1：在训练集上苦练 (做作业) ---
    permutation = np.random.permutation(num_samples)
    X_shuffled = X_train[:, permutation]
    Y_shuffled = Y_train[:, permutation]
    
    for i in range(0, num_samples, batch_size):
        X_batch = X_shuffled[:, i:i+batch_size]
        Y_batch = Y_shuffled[:, i:i+batch_size]
        train_batch_step(X_batch, Y_batch, learning_rate)
        
    # --- 阶段 2：在测试集上考试 (摸底测验) ---
    # 每学完一轮，就拉出它从来没见过的 10000 张图来考一下
    train_acc = evaluate(X_train, labels_train) # 看看作业做对多少
    test_acc = evaluate(X_test, labels_test)    # 看看真正考试拿多少分
    
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} | 耗时: {epoch_time:.2f}秒 | 作业准确率(Train): {train_acc:.2f}% | 真实考试成绩(Test): {test_acc:.2f}%")
    # 【新增】核心魔法：跑完一轮，缩小学习率！
    learning_rate = learning_rate * lr_decay
