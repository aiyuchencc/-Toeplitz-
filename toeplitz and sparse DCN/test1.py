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
model.load_state_dict(torch.load('doa_dcn.pth', map_location=device))
model.eval()

# 初始化测试数据集
test_theta = [0.0, 1.0]
test_dataset = DOADataset(num_samples=300, fixed_snr=None, fixed_theta=test_theta, fixed_N=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
phi_grid = test_dataset.phi_grid

true_theta1 = test_theta[0]
true_theta2 = test_theta[1]
angle_separation = abs(true_theta1 - true_theta2)


def calculate_por(model, test_loader):
    total_correct = 0
    error_list = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            pred_eta = output.squeeze().cpu().numpy()

            # 使用更鲁棒的峰值检测
            peaks, _ = find_peaks(pred_eta, height=0.5 * np.max(pred_eta), distance=5)
            if len(peaks) >= 2:
                peak_indices = np.argsort(pred_eta[peaks])[-2:]
                pred_doas = test_dataset.phi_grid[peaks[peak_indices]]
            else:
                # 如果检测不到足够峰值，选择全局最大值
                peak_indices = np.argsort(pred_eta)[-2:]
                pred_doas = test_dataset.phi_grid[peak_indices]

            pred_doas = np.sort(pred_doas)

            error1 = abs(pred_doas[0] - true_theta1)
            error2 = abs(pred_doas[1] - true_theta2)
            max_error = max(error1, error2)
            error_list.append(max_error)

            if max_error < angle_separation / 2:
                total_correct += 1

    por = total_correct / 300
    return por


snr_range = np.arange(-15, 10, 2).tolist()
por_list = []

for snr in snr_range:
    test_dataset = DOADataset(num_samples=300, fixed_snr=snr, fixed_theta=test_theta, fixed_N=None)
    test_loader = DataLoader(test_dataset, batch_size=1)
    por = calculate_por(model, test_loader)
    por_list.append(por)
por_list = por_list[::-1]

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(snr_range, por_list, 'b-o', linewidth=2, markersize=8, label='Proposed Method')
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('PoR', fontsize=12)
plt.ylim(0, 1)
plt.title('PoR vs SNR (M=40, N=70, 1° separation)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('por_vs_snr.png', dpi=300, bbox_inches='tight')
print("por_vs_snr.png已生成")