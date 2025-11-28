import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 临时允许重复加载
os.environ['OMP_NUM_THREADS'] = '1'  # 限制 OpenMP 线程
os.environ['MKL_NUM_THREADS'] = '1'  # 限制 MKL 线程
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================= 网络设计 =========================
class DCN(nn.Module):
    def __init__(self, input_len=121):
        super(DCN, self).__init__()
        self.net = nn.Sequential(
            # 输入形状: [batch, 2 (实虚部), L]
            nn.Conv1d(2, 24, kernel_size=21, padding=10),
            nn.BatchNorm1d(24),
            nn.ReLU(),

            nn.Conv1d(24, 20, kernel_size=15, padding=7),
            nn.BatchNorm1d(20),
            nn.ReLU(),

            nn.Conv1d(20, 12, kernel_size=11, padding=5),
            nn.BatchNorm1d(12),
            nn.ReLU(),

            nn.Conv1d(12, 5, kernel_size=5, padding=2),
            nn.BatchNorm1d(5),
            nn.ReLU(),

            nn.Conv1d(5, 1, kernel_size=3, padding=1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)
