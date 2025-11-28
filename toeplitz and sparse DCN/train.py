import torch
from torch import nn
from model import DCN
from torch.utils.data import TensorDataset, DataLoader, random_split

# ========================= 加载数据 =========================
data_path = 'doa_dataset.pt'
data = torch.load(data_path)
inputs = data['inputs']
labels = data['labels']

# 划分训练/验证集
train_size = int(0.8 * len(inputs))
val_size = len(inputs) - train_size
train_dataset, val_dataset = random_split(TensorDataset(inputs, labels), [train_size, val_size])

# 创建 DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

# ========================= 模型与训练设置 =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

epochs = 300
learning_rate = 0.001

# 初始化模型、损失函数和优化器
model = DCN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ========================= 训练循环 =========================
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        num_batches += 1

    train_loss = total_train_loss / num_batches if num_batches > 0 else 0.0

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

    val_loss /= len(val_loader)

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# ========================= 保存模型 =========================
torch.save(model.state_dict(), 'doa_dcn.pth')
print("网络权重参数已保存！")