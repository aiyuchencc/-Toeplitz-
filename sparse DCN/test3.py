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

# 初始化测试数据集
test_theta = [-3.33, -2.57]
test_dataset = DOADataset(num_samples=300, fixed_snr=None, fixed_theta=None ,fixed_N=None)  # 生成300个测试样本（蒙特卡洛试验）
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
phi_grid = test_dataset.phi_grid

true_theta1 = test_theta[0]
true_theta2 = test_theta[1]
angle_separation = abs(true_theta1 - true_theta2)


def calculate_rmse(model, test_loader):
    total_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # 模型预测
            data = data.to(device)  # 移动数据到GPU
            output = model(data)
            pred_eta = output.squeeze().cpu().numpy()  # 移回CPU再转numpy

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

            # RMSE累加
            total_error += (error1 ** 2 + error2 ** 2)
            num_samples += 2  # 每个样本贡献K=2个误差

    rmse = np.sqrt(total_error / 300)
    return rmse


snr_range = np.arange(-12, 12, 2).tolist()  # SNR从-12dB到10dB，步长2dB
rmse_list = []
for snr in snr_range:
    # 生成固定SNR固定theta的测试集
    test_dataset = DOADataset(num_samples=300, fixed_snr=snr, fixed_theta=test_theta, fixed_N=None)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # 计算RMSE
    rmse = calculate_rmse(model, test_loader)
    rmse_list.append(rmse)
rmse_list = rmse_list[::-1]
# 绘制rmse_vs_snr.png
plt.figure(figsize=(10, 6))
plt.plot(snr_range, rmse_list, 'r-o', linewidth=2, markersize=8, label='Proposed Method')
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('RMSE(degree)', fontsize=12)
plt.ylim(0, 4)
plt.title('RMSE vs SNR ', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('rmse_vs_snr.png', dpi=300, bbox_inches='tight')
print("rmse_vs_snr.png已生成")
