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

# 参数设置
true_theta1 = 0  # 固定第一个角度
true_theta2_range = np.arange(0.2, 3.1, 0.2).tolist()  # 角度分离范围0.2°到3.0°
snr = -8.5  # 固定SNR为-8.5dB
num_samples = 300  # 蒙特卡洛试验次数
M = 40  # 天线数
N = 70  # 样本数


def calculate_por(model, test_loader, angle_separation):
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
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
            # 计算最大角度误差
            error1 = abs(pred_doas[0] - true_theta1)
            error2 = abs(pred_doas[1] - true_theta2)
            max_error = max(error1, error2)

            # POR判断标准
            if max_error < angle_separation / 2:
                total_correct += 1

    por = total_correct / num_samples
    return por


# 计算不同角度分离下的POR
por_list = []
for true_theta2 in true_theta2_range:
    angle_separation = abs(true_theta1 - true_theta2)

    # 生成固定SNR和固定theta的测试集
    test_dataset = DOADataset(num_samples=num_samples,
                              fixed_snr=snr,
                              fixed_theta=[true_theta1, true_theta2],
                              fixed_N=N)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    por = calculate_por(model, test_loader, angle_separation)
    por_list.append(por)
    print(f"Angle separation: {angle_separation:.1f}°, PoR: {por:.3f}")

# # 确保POR单调递增
# for i in range(1, len(por_list)):
#     if por_list[i] < por_list[i - 1]:
#         por_list[i] = por_list[i - 1]

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(true_theta2_range, por_list, 'b-o', linewidth=2, markersize=8, label='Proposed Method')
plt.xlabel('Angle Separation (degree)', fontsize=12)
plt.ylabel('Probability of Resolution (PoR)', fontsize=12)
plt.title(f'PoR vs Angle Separation (M={M}, N={N}, SNR={snr}dB)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, 3.1, 0.5))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylim(0, 1.0)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('por_vs_angle_separation.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved as 'por_vs_angle_separation_monotonic.png'")