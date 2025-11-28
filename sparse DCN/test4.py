import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import DCN
from data import DOADataset
from scipy.signal import find_peaks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练模型
model = DCN().to(device)
model.load_state_dict(torch.load('doa_dcn1.pth', map_location=device))
model.eval()

# 初始化测试数据集的参数
test_theta = [-3.33, -2.57]
N_range = np.arange(50, 101, 5).tolist()  # 调整N的范围和步长
num_samples_per_N = 1000  # 增加每个N对应的测试样本数量
angle_separation = abs(test_theta[0] - test_theta[1])

# 定义计算RMSE的函数
def calculate_rmse(model, test_loader):
    total_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            pred_eta = output.squeeze().cpu().numpy()

            peaks, _ = find_peaks(pred_eta, height=0.5 * np.max(pred_eta), distance=5)
            if len(peaks) >= 2:
                peak_indices = np.argsort(pred_eta[peaks])[-2:]
                pred_doas = test_dataset.phi_grid[peaks[peak_indices]]
            else:
                # 如果检测不到足够峰值，选择全局最大值
                peak_indices = np.argsort(pred_eta)[-2:]
                pred_doas = test_dataset.phi_grid[peak_indices]

            pred_doas = np.sort(pred_doas)

            error1 = abs(pred_doas[0] - test_theta[0])
            error2 = abs(pred_doas[1] - test_theta[1])

            total_error += (error1 ** 2 + error2 ** 2)
            num_samples += 2

    rmse = np.sqrt(total_error / num_samples)
    return rmse

rmse_list = []

for N in N_range:
    # 生成固定SNR、固定theta和固定N的测试集
    test_dataset = DOADataset(num_samples=num_samples_per_N, fixed_snr=-8.5, fixed_theta=test_theta, fixed_N=N)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 计算RMSE
    rmse = calculate_rmse(model, test_loader)
    rmse_list.append(rmse)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(N_range, rmse_list, 'r-o', linewidth=2, markersize=8)
plt.xlabel('Number of Samples', fontsize=12)
plt.ylabel('RMSE (degree)', fontsize=12)
plt.title('RMSE vs N', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 0.5)  # 根据需要调整下限和上限
plt.savefig('rmse_vs_N.png', dpi=300, bbox_inches='tight')
print("rmse_vs_N.png已生成")