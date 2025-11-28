# ========== 生成并保存数据 ==========
import torch
from data import DOADataset

num_samples = 43200
batch_size = 64
print("生成数据中...")
dataset = DOADataset(num_samples)

inputs = []
labels = []
for i in range(len(dataset)):
    x, y = dataset[i]
    inputs.append(x)
    labels.append(y)


inputs_tensor = torch.stack(inputs)
labels_tensor = torch.stack(labels)

# 保存为 .pt 文件
torch.save({'inputs': inputs_tensor, 'labels': labels_tensor}, 'doa_dataset1.pt')
print("数据保存完成：doa_dataset1.pt")

# ========== 加载并构建 DataLoader ==========
# print("加载数据中...")
# data = torch.load('doa_dataset.pt')
# inputs = data['inputs']
# labels = data['labels']
#
# full_dataset = TensorDataset(inputs, labels)
#
# train_size = int(0.8 * len(full_dataset))
# val_size = len(full_dataset) - train_size
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
#
# print("数据加载完成，训练样本数：", len(train_dataset), "验证样本数：", len(val_dataset))