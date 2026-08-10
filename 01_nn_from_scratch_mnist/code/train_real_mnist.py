import numpy as np
from datasets import load_dataset
import time

# ==========================================
# 1. 加载并预处理真实数据
# ==========================================
print("正在从本地加载 MNIST 数据集...")
# ds = load_dataset("ylecun/mnist", data_dir="./data") # 优先从本地缓存读取
ds = load_dataset("ylecun/mnist") # 优先从本地缓存读取

# 提取训练集数据
print("正在将图片转换为 NumPy 数组并进行归一化...")
# 将 60000 张图片转换为 (60000, 784, 1) 的三维数组，并归一化
train_images = np.array([np.array(img) for img in ds['train']['image']])
X_train = train_images.reshape(-1, 784, 1).astype('float32') / 255.0

# 提取标签并进行 One-Hot 编码
print("正在对标签进行 One-Hot 编码...")
labels = np.array(ds['train']['label'])
Y_train = np.zeros((len(labels), 10, 1))
for i, label in enumerate(labels):
    Y_train[i, label, 0] = 1.0  # 对应数字的位置设为 1，其他为 0

print(f"数据准备完毕！X_train 形状: {X_train.shape}, Y_train 形状: {Y_train.shape}\n")

# ==========================================
# 2. 神经网络核心组件 (完全复用手推公式)
# ==========================================
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x)) 
    return exp_x / np.sum(exp_x)

# 初始化权重和偏置 (维度严格对齐我们的手稿)
W0 = np.random.randn(16, 784) * 0.01
b0 = np.zeros((16, 1))

W1 = np.random.randn(15, 16) * 0.01
b1 = np.zeros((15, 1))

W2 = np.random.randn(10, 15) * 0.01
b2 = np.zeros((10, 1))

# ==========================================
# 3. 核心前向与反向传播算法
# ==========================================
def train_step(X, Y_onehot, learning_rate=0.01):
    global W0, b0, W1, b1, W2, b2
    
    # --- 前向传播 ---
    h1 = np.dot(W0, X) + b0
    a1 = relu(h1)
    
    h2 = np.dot(W1, a1) + b1
    a2 = relu(h2)
    
    h3 = np.dot(W2, a2) + b2
    p = softmax(h3)
    
    # 记录损失函数 (交叉熵)
    loss = -np.sum(Y_onehot * np.log(p + 1e-8))
    
    # --- 反向传播 ---
    g3 = p - Y_onehot
    dW2 = np.dot(g3, a2.T)
    db2 = g3
    
    g2 = np.dot(W2.T, g3) * relu_deriv(h2)
    dW1 = np.dot(g2, a1.T)
    db1 = g2
    
    g1 = np.dot(W1.T, g2) * relu_deriv(h1)
    dW0 = np.dot(g1, X.T)
    db0 = g1
    
    # --- 参数更新 ---
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W0 -= learning_rate * dW0
    b0 -= learning_rate * db0
    
    # 返回当前的损失，以及网络预测概率最高的那个数字
    predicted_digit = np.argmax(p) 
    return loss, predicted_digit

# ==========================================
# 4. 开始真实训练！(Training Loop)
# ==========================================
epochs = 1  # 为了演示，我们先只让网络看全部 6 万张图 1 遍
learning_rate = 0.01
num_samples = len(X_train)

print(f"=== 开始训练 (共 {epochs} Epochs, {num_samples} 张图片) ===")
start_time = time.time()
j = 0

for epoch in range(epochs):
    correct_predictions = 0
    running_loss = 0.0
    
    for i in range(num_samples):
        # 抓取第 i 张图片和它的真实标签
        x_current = X_train[i] # shape [784, 1]
        y_current = Y_train[i] # shape [10, 1]

        if j == 0 :
            print(f"x_current shape: {x_current.shape}, {y_current.shape}\n")
            j = 1
        
        
        # 丢进网络训练一次
        loss, predicted_digit = train_step(x_current, y_current, learning_rate)
        
        # 统计数据
        running_loss += loss
        if predicted_digit == labels[i]:
            correct_predictions += 1
            
        # 每看 5000 张图，汇报一次战况
        if (i + 1) % 5000 == 0:
            accuracy = (correct_predictions / 5000) * 100
            avg_loss = running_loss / 5000
            print(f"已训练 {i+1}/60000 张图片 | 平均 Loss: {avg_loss:.4f} | 当前准确率: {accuracy:.2f}%")
            
            # 清零统计，重新计算接下来的 5000 张
            correct_predictions = 0
            running_loss = 0.0

end_time = time.time()
print(f"\n=== 训练完成！耗时: {end_time - start_time:.2f} 秒 ===")

# ==========================================
# 5. 随便抽一张图考考它
# ==========================================
test_index = np.random.randint(0, len(X_train)) # 随机抽一张
test_img = X_train[test_index]
true_label = labels[test_index]

# 走一遍前向传播
h1 = np.dot(W0, test_img) + b0
a1 = relu(h1)
h2 = np.dot(W1, a1) + b1
a2 = relu(h2)
h3 = np.dot(W2, a2) + b2
p = softmax(h3)

print(f"\n[考试时间] 我们随机抽取了第 {test_index} 张图片：")
print(f"这张图真实写的是数字: {true_label}")
print(f"你的 AI 预测它是数字: {np.argmax(p)}")
print(f"AI 有多确定？ {np.max(p)*100:.2f}% 的把握！")
