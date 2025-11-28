% 参数设置
M = 40;      % 阵元数
N = 70;      % 快拍数
K = 2;       % 信号源数
lambda = 1;  % 波长
d = lambda/2; % 阵元间距
theta_true = [0,1];    % 真实DOA（角度间隔1）
SNR_range = -15:2:9;   % SNR范围（dB）
num_trials = 300;       % 蒙特卡洛试验次数
angle_grid = -6:0.1:6;% 角度搜索范围

% 初始化PoR结果
PoR_results = zeros(length(SNR_range), 1);
PoR_results_shrink = zeros(length(SNR_range), 1);
% 主仿真循环
for idx = 1:length(SNR_range)
    SNR_dB = SNR_range(idx);
    num_resolved = 0;
    num_resolved1 = 0;
    for trial = 1:num_trials
        % 生成信号和噪声
        A = exp(-1j * 2 * pi * d/lambda * (0:M-1)' * sind(theta_true));%40*2
        S = (randn(K, N) + 1j * randn(K, N)) / sqrt(2);%2*70
       % noise_power = 10^(-SNR_dB/10);%将dB转换为十进制
       % noise = sqrt(noise_power/2) * (randn(M, N) + 1j * randn(M, N));
       % Y = A * S + noise;
        Y = A * S; %40*70
        Y=awgn(Y,SNR_dB,'measured');
        
        % 调用MUSIC算法
       [spectrum_scm,spectrum_toeplitz,estimated_doas_scm,estimated_doas_toeplitz] = music_test(Y,N,M,K,angle_grid);
        
        % 判断是否成功分辨
        if length(estimated_doas_scm) == 2
            error1 = abs(estimated_doas_scm(1) - theta_true(1));
            error2 = abs(estimated_doas_scm(2) - theta_true(2));
            max_error = max(error1, error2);
            angle_separation = abs(theta_true(1) - theta_true(2));
            if max_error <= angle_separation/2  
                num_resolved = num_resolved + 1;
            end
        end
  
        if length(estimated_doas_toeplitz) == 2
            error11 = abs(estimated_doas_toeplitz(1) - theta_true(1));
            error22 = abs(estimated_doas_toeplitz(2) - theta_true(2));
            max_error1 = max(error11, error22);
            angle_separation1 = abs(theta_true(1) - theta_true(2));
            if max_error1 <=angle_separation1/2  
                num_resolved1 = num_resolved1 + 1;
            end
         end
    end

    PoR_results(idx) = num_resolved / num_trials;
    PoR_results_shrink(idx) = num_resolved1 / num_trials;
end
% 绘图
figure;
plot(SNR_range, PoR_results, 'bs-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
hold on;
plot(SNR_range, PoR_results_shrink, 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
hold off;
xlabel('SNR (dB)');
ylabel('PoR');
xlim([-15, 9]);
ylim([0, 1]);
title('PoR vs SNR (\Delta\theta=1^\circ)');
legend('MUSIC', 'MUSICToeplitz');
grid on; 

