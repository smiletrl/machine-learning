import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets  # 引入 torchvision 用于下载数据
import time
import swanlab

# ==========================================
# 1. 加载并打包数据 (引入测试集)
# ==========================================
print("正在通过 PyTorch (torchvision) 下载/加载数据集...")

# 使用 torchvision.datasets.MNIST 自动下载并加载数据
# 参数 root='./data' 表示数据将保存在当前目录下的 data 文件夹内
train_data_raw = datasets.MNIST(root='./data', train=True, download=True)
test_data_raw = datasets.MNIST(root='./data', train=False, download=True)

# 【PyTorch 特色 1：张量形状】
# 原数据格式为 (60000, 28, 28) 的 uint8 像素矩阵。
# 这里用 .view(-1, 784) 直接将 28x28 展平为 784，并将 uint8 转为 float32 后归一化。
# --- 准备训练集 (日常作业，60000张) ---
X_train = train_data_raw.data.view(-1, 784).float() / 255.0

# 【PyTorch 特色 2：告别 One-Hot】
# PyTorch 算交叉熵时，底层会自动处理类别标签，直接传数字即可！
# torchvision 返回的 targets 已经是 torch.long 类型的张量
Y_train = train_data_raw.targets

# --- 准备测试集 (期末考试，10000张，完全不参与训练) ---
print("正在准备期末考试卷(测试集)...")
X_test = test_data_raw.data.view(-1, 784).float() / 255.0
Y_test = test_data_raw.targets

# 【PyTorch 特色 3：DataLoader 数据管家】
# 替代原本手写的 permutation 和 for 循环切片，它会自动帮我们打乱数据并切分成 Batch。
batch_size = 64
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 准备一个字典，用来临时存放钩子抓取到的激活值
activation_cache = {}

# 定义一个生产钩子的工厂函数
def get_activation(layer_name):
    def hook(module, input, output):
        # 把这一层的输出（激活值）剥离计算图后，存入字典
        activation_cache[layer_name] = output.detach()
    return hook

# ==========================================
# 2. 神经网络组件
# ==========================================
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch 把权重初始化（W 和 b）自动藏在了 nn.Linear 里
        # 结构与手写的一致：784 -> 16 -> 15 -> 10
        self.net = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 15),
            nn.ReLU(),
            nn.Linear(15, 10)
            # 【注意】PyTorch 的 CrossEntropyLoss 自带 Softmax 效果，
            # 所以这里最后一层不需要单独加 Softmax 层！输出的是纯净的“对数几率 (Logits)”。
        )

    def forward(self, x):
        return self.net(x)

# 实例化网络
model = MNISTNet()

# 将钩子挂在 model.net 的第 1 层（即第一个 ReLU 层）上
# 注意：索引 0 是 Linear(784, 16)，索引 1 是 ReLU()
model.net[1].register_forward_hook(get_activation('relu_1'))

swanlab.init(
    project="MNIST-PyTorch-Hardcore",
    experiment_name="SGD_base_Model",
    config={
        "learning_rate": 0.1,
        "batch_size": 64,
        "epochs": 5,
        "optimizer": "SGD"
    }
)

# ==========================================
# 3. 核心算法 (定义损失函数、优化器和学习率策略)
# ==========================================
epochs = 5
learning_rate = 0.5
lr_decay = 0.8

# 相当于手动算的交叉熵误差 (内部包含 Softmax)
criterion = nn.CrossEntropyLoss()

# 优化器：负责更新 W 和 b
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 学习率调度器：跑完一轮，步子缩小
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)


# ==========================================
# 4. 训练与测试循环
# ==========================================
def evaluate(X, Y):
    """考试机制：只做题不看答案，不更新权重。测试真正的泛化能力。"""
    model.eval()               # 告诉网络：现在是考试状态（关掉训练特有的机制）
    with torch.no_grad():      # 告诉 PyTorch：不用在后台记账算梯度了，省点内存和算力
        logits = model(X)      # 一次性让全量数据穿过网络
        predictions = torch.argmax(logits, dim=1) # 选出概率最大的数字
        accuracy = (predictions == Y).float().mean() * 100
    return accuracy.item()

print(f"\n=== 开始 PyTorch 硬核实战 (Batch Size: {batch_size}) ===")
for epoch in range(epochs):
    start_time = time.time()
    
    # --- 阶段 1：在训练集上苦练 (做作业) ---
    model.train()              # 告诉网络：现在是做作业状态，准备学习
    for step, (X_batch, Y_batch) in enumerate(train_loader):
        
        # 1. 归零脑回路里的残余记忆 (清空上一轮的梯度)
        optimizer.zero_grad()
        
        # 2. 前向传播：看题、给出预测
        outputs = model(X_batch)
        
        # 3. 计算误差：对比标准答案
        loss = criterion(outputs, Y_batch)

        # 4. 反向传播：纠正脑回路 (【核心】PyTorch 自动算 dW 和 db！)
        loss.backward()
        
        # 5. 参数更新：真正把知识记在脑子里
        optimizer.step()

        if step % 100 == 0:
            current_step = epoch * len(train_loader) + step
            swanlab.log({"training_loss": loss.item()}, step=current_step)

            # swanlab.log({"weights/layer1": model.net[0].weight.detach()}, step=current_step)

            weight_tensor = model.net[0].weight.detach().cpu().numpy()

            # 提取关键的统计特征，用极其轻量的折线图来追踪
            swanlab.log({
                "weights_layer1/mean": weight_tensor.mean().item(),  # 看中心点是否发生偏移
                "weights_layer1/std": weight_tensor.std().item(),    # 看分布有多宽 (如果趋近0说明全变成一样的死数了)
                "weights_layer1/max": weight_tensor.max().item(),    # 抓“爆炸”的离群点
                "weights_layer1/min": weight_tensor.min().item(),    # 抓“异常”的下限
            }, step=current_step)
            
            act_tensor = activation_cache['relu_1']
            swanlab.log({
                "activations_relu1/mean": act_tensor.float().mean().item(),
                "activations_relu1/max": act_tensor.float().max().item(),
                "activations_relu1/dead_ratio": (act_tensor == 0).float().mean().item() # 监控死神经元比例
            }, step=current_step)

             # 记录激活值的直方图
            # act = activation_cache['relu_1'].detach().cpu().numpy()
            # swanlab.log({"activations/relu1_hist": swanlab.Histogram(act.flatten())}, step=current_step)

    # --- 阶段 2：在测试集上考试 (摸底测验) ---
    train_acc = evaluate(X_train, Y_train) # 看看作业做对多少
    test_acc = evaluate(X_test, Y_test)    # 看看真正考试拿多少分
    
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} | 耗时: {epoch_time:.2f}秒 | 作业准确率(Train): {train_acc:.2f}% | 真实考试成绩(Test): {test_acc:.2f}%")
    
    # 【核心魔法】：跑完一轮，步子缩小到原来的 80%
    scheduler.step()

swanlab.finish()
