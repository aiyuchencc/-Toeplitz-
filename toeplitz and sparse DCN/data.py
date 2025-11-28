import os
import torch
import numpy as np
from torch.utils.data import Dataset

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
M = 40
L = 121
K = 2
SNR_ranges = [(-10, 0), (0, 10)]
lamb = 1
d = lamb / 2
class DOADataset(Dataset):
    def __init__(self, num_samples, fixed_snr=None, fixed_theta=None, fixed_N=None):
        self.num_samples = num_samples
        self.phi_grid = np.linspace(-6, 6, L)
        self.fixed_snr = fixed_snr
        self.fixed_theta = fixed_theta
        self.fixed_N = fixed_N

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = 70

        if self.fixed_theta is not None:
            theta = np.array(self.fixed_theta, dtype=np.float32)
        else:
            theta = np.array(self.generate_theta(K, 0.2, -6, 6), dtype=np.float32)


        if self.fixed_snr is not None:
            snr = self.fixed_snr
        else:
            snr_range = np.random.choice([0, 1], p=[2 / 3, 1 / 3])
            snr = np.random.uniform(*SNR_ranges[snr_range])

        s = (np.random.randn(K, N) + 1j * np.random.randn(K, N)) / np.sqrt(2)
        A = np.exp(-1j * 2 * d / lamb * np.pi * np.sin(theta * np.pi / 180) * np.arange(M)[:, None])
        noise_power = 10 ** (-snr / 10)
        noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(noise_power / 2)
        y = A @ s + noise
        R_hat = (y @ y.conj().T) / N
        R_T = self.toeplitz_rectification(R_hat)
        alpha = self.calculate_alpha(R_hat, R_T, N)
        R_bar = (1 - alpha) * R_hat + alpha * R_T
        Phi = self.build_overcomplete_basis()
        r = R_bar.reshape(-1, 1)
        eta_tilde = Phi.conj().T @ r

        eta = np.zeros(L)
        for theta_k in theta:
            idx = np.argmin(np.abs(self.phi_grid - theta_k))  # 找到方向网格中最接近的网格点
            eta[idx] = 1

        eta_tilde_real_imag = np.stack([eta_tilde.real.squeeze(), eta_tilde.imag.squeeze()],
                                       axis=0)  # 将 eta_tilde 分解为实部和虚部
        input_data = torch.tensor(eta_tilde_real_imag, dtype=torch.float32)
        label = torch.tensor(eta, dtype=torch.float32)
        return input_data, label

    def toeplitz_rectification(self, R):
        R_tensor = torch.tensor(R, dtype=torch.complex64)
        M = R_tensor.shape[0]
        R_T = torch.zeros_like(R_tensor)
        for m in range(-M + 1, M):
            J_m = torch.eye(M, dtype=R_tensor.dtype)
            if m != 0:
                J_m = torch.roll(J_m, shifts=m, dims=1)
            trace_val = torch.trace(R_tensor @ J_m)
            avg = trace_val / (M - abs(m))
            J_neg_m = J_m.T.conj()
            R_T += avg * J_neg_m
        return R_T.numpy()

    def calculate_alpha(self, R_hat, R_T, N):
        numerator = (N - 3) * np.trace(R_hat @ R_hat) + (N - 1) * np.trace(R_hat) ** 2
        denominator = (N - 2) * (N + 1) * np.trace((R_hat - R_T) @ (R_hat - R_T).conj().T)
        alpha_hat = numerator / denominator
        return np.minimum(alpha_hat, 1)

    def build_overcomplete_basis(self):
        Phi = np.zeros((M ** 2, L), dtype=complex)
        for l in range(L):
            a = np.exp(-1j * np.pi * np.sin(self.phi_grid[l] * np.pi / 180) * np.arange(M))
            Phi[:, l] = np.outer(a, a.conj()).flatten()
        return Phi

    def generate_theta(self, K=2, min_sep=0.2, min_val=-6, max_val=6):
        """
        生成K个在[min_val, max_val]范围内的随机角度，确保相邻角度之间的间隔至少为min_sep。
        """
        thetas = []
        for _ in range(K):
            # 随机生成一个角度
            theta = np.random.uniform(min_val, max_val)
            # 如果已经生成了角度，检查间隔是否满足条件
            if thetas:
                # 找到最接近当前角度的已生成角度
                closest = min(thetas, key=lambda x: abs(x - theta))
                # 如果间隔小于最小间隔，重新生成
                while abs(theta - closest) < min_sep:
                    theta = np.random.uniform(min_val, max_val)
                    if theta:
                        closest = min(thetas, key=lambda x: abs(x - theta))
            thetas.append(theta)
        # 排序并返回
        return sorted(thetas)