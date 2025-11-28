% 参数设置
M = 40;      % 阵元数
N = 70;      % 快拍数
K = 2;       % 信号源数
lambda = 1;  % 波长
d = lambda/2; % 阵元间距
theta_true = [0,1]; % 真实DOA角度
angle_grid = -6:0.1:6;%角度范围

% 生成接收信号
A = exp(-1j * 2 * pi * d/lambda * (0:M-1)' * sin(theta_true * pi/180));%导向矢量矩阵
S = randn(K, N) + 1j * randn(K, N); % 信号源 randn生成标准高斯分布
noise = 0.1 * (randn(M, N) + 1j * randn(M, N)); % 噪声  
Y = A * S + noise;%信噪比20dB  SNR（dB）=20lgS/N

% 调用music_test算法
[spectrum_scm,spectrum_toeplitz,estimated_doas_scm,estimated_doas_toeplitz] = music_test(Y,N,M,K,angle_grid);

%显示角度值
disp(['MUSIC估计角度: ', num2str(estimated_doas_scm)]);
disp(['收缩后MUSIC估计角度: ', num2str(estimated_doas_toeplitz)]);

% 绘制对比图
figure;
plot(angle_grid, 10*log10(abs(spectrum_scm)/max(abs(spectrum_scm))), 'b-', 'LineWidth', 1.5);
%hold on;
%plot(angle_grid, 10*log10(abs(spectrum_toeplitz)/max(abs(spectrum_toeplitz))), 'r-', 'LineWidth', 1.5);
%hold off;

xlabel('角度 (度)');
ylabel('归一化空间谱 (dB)');
title('MUSIC算法对比');
legend('直接SCM', '托普利兹正则化');
grid on;



