# 📡 DOA 估计系统 - 基于深度学习和传统MUSIC算法

## 项目简介

本项目实现了波达方向(DOA, Direction of Arrival)估计的深度学习解决方案(DCN网络)与传统MUSIC算法的对比系统。包含完整的数据生成、模型训练、测试评估流程，可用于阵列信号处理研究。

本项目是毕业设计《基于Toeplitz和稀疏先验的深度学习波达角度估计方法研究》的完整代码实现。

## 功能特性

✔️ **支持四种DOA估计方法：**
- 传统MUSIC算法
- 基于Toeplitz先验MUSIC算法
- 基于Toeplitz和稀疏先验深度卷积网络(DCN)
- 基于稀疏先验深度卷积网络(DCN)

✔️ **完整的仿真数据生成管道**

✔️ **自动化训练与评估框架**

✔️ **可视化结果对比** (PoR-SNR曲线、PoR-Angle Separation曲线、RMSE分析等)

## 项目结构

```
.
├── MUSIC/                          # MATLAB平台下的MUSIC算法实现
│   ├── main.m                      # 算法有效性测试主文件
│   ├── music_test.m                # MUSIC算法核心实现
│   ├── PoR_SNR.m                   # SNR与正确率(PoR)关系测试
│   ├── PoR_AngleSeperation.m       # 角度分离度与PoR关系测试
│   ├── linear_shrinkage.m          # 线性收缩估计
│   └── toeplitz_prior.m            # Toeplitz先验处理
│
├── toeplitz and sparse DCN/        # 基于Toeplitz和稀疏先验的DCN网络
│   ├── data.py                     # 数据生成类(包含Toeplitz正则化)
│   ├── dataset.py                  # 数据集生成脚本
│   ├── model.py                    # DCN网络模型定义
│   ├── train.py                    # 模型训练脚本
│   ├── test1.py                    # 测试1: PoR vs SNR
│   ├── test2.py                    # 测试2: PoR vs Angle Separation
│   ├── test3.py                    # 测试3: RMSE vs SNR
│   ├── test4.py                    # 测试4: RMSE vs N(快拍数)
│   ├── doa_dataset.pt              # 训练数据集
│   └── doa_dcn.pth                 # 训练好的模型权重
│
├── sparse DCN/                     # 仅基于稀疏先验的DCN网络
│   ├── data.py                     # 数据生成类(无Toeplitz正则化)
│   ├── dataset.py                  # 数据集生成脚本
│   ├── model.py                    # DCN网络模型定义
│   ├── train.py                    # 模型训练脚本
│   ├── test1.py                    # 测试1: PoR vs SNR
│   ├── test2.py                    # 测试2: PoR vs Angle Separation
│   ├── test3.py                    # 测试3: RMSE vs SNR
│   ├── test4.py                    # 测试4: RMSE vs N(快拍数)
│   ├── doa_dataset1.pt             # 训练数据集
│   └── doa_dcn1.pth                # 训练好的模型权重
│
├── 仿真图片结果/                    # 实验结果可视化
│   ├── combined_*.png              # 算法对比分析图
│   ├── Dataset *.csv               # 提取的实验数据
│   ├── hebing_*.py                 # 数据合并与绘图脚本
│   └── *.bmp, *.vsdx               # 网络结构图和流程图
│
└── README.md                        # 本文件
```

## 环境要求

### Python环境
- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- SciPy
- CUDA 11.3+ (GPU训练推荐，可选)

### MATLAB环境
- MATLAB R2016a+

### 安装依赖

```bash
# 安装Python依赖
pip install torch numpy matplotlib scipy

# 如果使用GPU，请根据CUDA版本安装对应的PyTorch
# 例如：pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

## 快速开始

### 1. MUSIC算法 (MATLAB)

MUSIC文件夹内为MATLAB平台下的MUSIC算法和基于Toeplitz先验MUSIC算法的仿真。

#### 代码文件关系图

```
PoR_SNR.m / PoR_AngleSeperation.m
  │
  ├─▶ music_test.m
  │     │
  │     ├─▶ linear_shrinkage.m
  │     │     └─▶ toeplitz_prior.m
  │     │
  │     └─▶ 原生MUSIC处理
  │
  └─▶ 结果分析和绘图
```

#### 使用示例

```matlab
% 运行主测试文件
run('MUSIC/main.m')

% 运行SNR与PoR关系测试
run('MUSIC/PoR_SNR.m')

% 运行角度分离度与PoR关系测试
run('MUSIC/PoR_AngleSeperation.m')
```

### 2. Toeplitz和稀疏先验DCN网络 (Python)

#### 典型执行流程

**步骤1: 生成数据集**

```bash
cd "toeplitz and sparse DCN"
python dataset.py
# 输出：doa_dataset.pt
```

**步骤2: 训练模型**

```bash
python train.py
# 输出：doa_dcn.pth
# 训练300个epoch，自动划分80%训练集和20%验证集
```

**步骤3: 测试评估**

```bash
# 测试PoR vs SNR
python test1.py
# 输出：por_vs_snr.png

# 测试PoR vs Angle Separation
python test2.py
# 输出：por_vs_angle_separation.png

# 测试RMSE vs SNR
python test3.py
# 输出：rmse_vs_snr.png

# 测试RMSE vs N(快拍数)
python test4.py
# 输出：rmse_vs_N.png
```

#### 代码文件关系图

```
1. 数据生成阶段：
dataset.py 作为入口脚本
  └─▶ 调用 data.py 中的 DOADataset 类
      └─▶ 生成并保存Tensor格式数据集到 doa_dataset.pt

2. 模型训练阶段：
train.py
  ├─ 加载 doa_dataset.pt
  ├─ 初始化 DCN 模型 (来自 model.py)
  ├─ 进行300轮训练迭代
  │   ├─ 前向传播
  │   ├─ 计算MSE损失
  │   └─ 反向传播更新权重
  └─ 保存训练好的模型 doa_dcn.pth

3. 测试评估阶段：
test1.py / test2.py / test3.py / test4.py
  ├─ 加载 doa_dcn.pth
  ├─ 调用 DOADataset 生成测试数据
  ├─ 对每个测试条件:
  │   ├─ 计算模型输出
  │   ├─ 峰值检测确定DOA
  │   └─ 统计正确率(PoR)或RMSE
  └─ 绘制并保存结果曲线图
```

### 3. 仅稀疏先验DCN网络 (Python)

该文件夹内的代码与"toeplitz and sparse DCN"文件夹基本相同，**唯一区别**在于`data.py`中删去了Toeplitz正则化过程，仅使用稀疏先验。

#### 典型执行流程

```bash
cd "sparse DCN"

# 1. 生成数据
python dataset.py
# 输出：doa_dataset1.pt

# 2. 训练模型
python train.py
# 输出：doa_dcn1.pth

# 3. 测试评估
python test1.py
# 输出：por_vs_snr.png
```

## 算法说明

### DCN网络架构

DCN (Deep Convolutional Network) 是一个一维卷积神经网络，用于从稀疏表示中恢复DOA信息。

**网络结构：**

```python
nn.Sequential(
    # 输入: [batch, 2 (实虚部), 121]
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
    # 输出: [batch, 121] (角度网格上的概率分布)
)
```

**关键参数：**
- 输入维度: 2×121 (实部+虚部，121个角度网格点)
- 角度范围: -6° 到 6°
- 角度分辨率: 0.1°
- 输出维度: 121 (每个角度网格点的激活值)

### MUSIC算法流程

1. **计算样本协方差矩阵** (Sample Covariance Matrix, SCM)
2. **特征分解获取噪声子空间**
3. **构建空间谱函数**
4. **谱峰搜索确定DOA**

### Toeplitz正则化

在"toeplitz and sparse DCN"方法中，使用Toeplitz先验对样本协方差矩阵进行正则化：

1. 计算样本协方差矩阵 R̂
2. 通过Toeplitz校正得到 R_T
3. 计算收缩系数 α
4. 得到正则化协方差矩阵: R̄ = (1-α)R̂ + αR_T

### 稀疏表示

使用过完备基(Overcomplete Basis)将协方差矩阵向量化后投影到稀疏域：

- 构建过完备基矩阵 Φ (维度: M² × L)
- 计算稀疏表示: η̃ = Φ^H · r
- 输入到DCN网络进行DOA估计

## 实验参数设置

### 阵列参数
- **阵元数 (M)**: 40
- **阵元间距 (d)**: λ/2 (半波长)
- **信号源数 (K)**: 2
- **快拍数 (N)**: 70 (默认)

### 训练参数
- **训练轮数 (epochs)**: 300
- **学习率 (learning_rate)**: 0.001
- **优化器**: Adam
- **损失函数**: MSE Loss
- **批次大小 (batch_size)**: 64
- **训练/验证集划分**: 80% / 20%

### 数据生成参数
- **SNR范围**: -10 dB 到 10 dB
- **角度范围**: -6° 到 6°
- **最小角度间隔**: 0.2°
- **角度网格数 (L)**: 121

## 实验结果

完整实验结果与分析请参考毕业论文《基于Toeplitz和稀疏先验的深度学习波达角度估计方法研究》。

### 性能指标

- **PoR (Probability of Resolution)**: 正确分辨率，衡量DOA估计的准确性
- **RMSE (Root Mean Square Error)**: 均方根误差，衡量估计角度的精度

### 实验结果文件

实验结果保存在`仿真图片结果/`文件夹中，包括：

- `combined_snr.png`: 各算法PoR vs SNR对比
- `combined_angle_separation.png`: 各算法PoR vs 角度分离度对比
- `combined_rmse_snr.png`: 各算法RMSE vs SNR对比
- `combined_rmse_N.png`: 各算法RMSE vs 快拍数对比

## 文件说明

### 核心代码文件

#### `data.py`
- **功能**: 定义`DOADataset`类，负责生成DOA估计的训练和测试数据
- **关键方法**:
  - `__getitem__()`: 生成单个数据样本
  - `toeplitz_rectification()`: Toeplitz校正(仅toeplitz版本)
  - `calculate_alpha()`: 计算收缩系数
  - `build_overcomplete_basis()`: 构建过完备基矩阵
  - `generate_theta()`: 生成满足最小间隔要求的随机角度

#### `model.py`
- **功能**: 定义DCN网络模型架构
- **类**: `DCN(nn.Module)`

#### `train.py`
- **功能**: 模型训练主脚本
- **流程**: 数据加载 → 模型初始化 → 训练循环 → 模型保存

#### `test1.py` - `test4.py`
- **test1.py**: PoR vs SNR 性能测试
- **test2.py**: PoR vs Angle Separation 性能测试
- **test3.py**: RMSE vs SNR 性能测试
- **test4.py**: RMSE vs N(快拍数) 性能测试

## 使用建议

1. **首次运行**: 建议先运行`dataset.py`生成数据集，然后运行`train.py`训练模型
2. **GPU加速**: 如果有GPU，训练速度会显著提升，代码会自动检测并使用GPU
3. **参数调整**: 可以在`data.py`和`train.py`中调整实验参数(SNR范围、角度范围、训练轮数等)
4. **结果对比**: 运行不同文件夹下的测试脚本，对比Toeplitz版本和仅稀疏版本的性能差异

## 常见问题

### Q: 训练时出现内存不足错误？
A: 可以减小`batch_size`或`num_samples`参数。

### Q: 如何修改角度范围？
A: 在`data.py`中修改`phi_grid = np.linspace(-6, 6, L)`，并相应调整`generate_theta()`中的`min_val`和`max_val`参数。

### Q: 如何修改阵元数？
A: 在`data.py`中修改`M = 40`，注意需要重新生成数据集和训练模型。

### Q: 测试结果与论文不一致？
A: 确保使用相同的随机种子，或增加测试样本数量以获得更稳定的统计结果。

## 引用

如果本项目对您的研究有帮助，请引用：

```
基于Toeplitz和稀疏先验的深度学习波达角度估计方法研究
```

## 许可证

本项目为毕业设计代码，仅供学习和研究使用。

## 联系方式zjzsyyc0580@163.com

如有问题或建议，欢迎通过以下方式联系：
- 提交Issue
- 发送邮件

---

**注意**: 本项目代码包含详细注释，便于理解和复现实验结果。

