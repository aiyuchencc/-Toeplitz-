% 参数设置
M = 40;      % 阵元数
N = 70;      % 快拍数
K = 2;       % 信号源数
lambda = 1;  % 波长
d = lambda/2; % 阵元间距
SNR_dB = -8.5;             % 固定信噪比
angle_separations = 0.2:0.2:3; % 角度分离范围（度）
num_trials = 300;       % 蒙特卡洛试验次数
angle_grid = -6:0.1:6;% 角度搜索范围

% 初始化PoR结果
PoR_results = zeros(length(angle_separations), 1);
PoR_results1 = zeros(length(angle_separations), 1);

% 主仿真循环
for idx = 1:length(angle_separations)
    delta_theta = angle_separations(idx);
    theta_true = [0, delta_theta]; % 固定θ1=0°，θ2=delta_theta
    num_resolved = 0;
    num_resolved1 = 0;
    for trial = 1:num_trials
        % 生成信号和噪声
        A = exp(-1j * 2 * pi * d/lambda * (0:M-1)' * sind(theta_true));
        S = (randn(K, N) + 1j * randn(K, N)) / sqrt(2);
        %noise_power = 10^(-SNR_dB/10);
        %noise = sqrt(noise_power/2) * (randn(M, N) + 1j * randn(M, N));
        %Y = A * S + noise;
        Y = A * S;
        Y=awgn(Y,SNR_dB,'measured');
        % 调用MUSIC算法
        [spectrum_scm,spectrum_toeplitz,estimated_doas_scm,estimated_doas_toeplitz] = music_test(Y,N,M,K,angle_grid);
        
        % 判断是否成功分辨
        if length(estimated_doas_scm) >= 2
            error1 = abs(estimated_doas_scm(1) - theta_true(1));
            error2 = abs(estimated_doas_scm(2) - theta_true(2));
            max_error = max(error1, error2);
            if max_error <= delta_theta/2 
                num_resolved = num_resolved + 1;
            end
        end
   
     if length(estimated_doas_toeplitz) >= 2
            error11 = abs(estimated_doas_toeplitz(1) - theta_true(1));
            error22 = abs(estimated_doas_toeplitz(2) - theta_true(2));
            max_error1 = max(error11, error22);
            if max_error1 <= delta_theta/2
                num_resolved1 = num_resolved1 + 1;
            end
     end
    end
    PoR_results(idx) = num_resolved / num_trials;
    PoR_results1(idx) = num_resolved1 / num_trials;
end

% 绘图
figure;
plot(angle_separations, PoR_results, 'bs-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
hold on;
plot(angle_separations, PoR_results1, 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
hold off;
xlabel('Angle Separation (degrees)'); 
ylabel('PoR');
xlim([0.2, 3]);
ylim([0, 1]);
title(['PoR vs Angle Separation (SNR = ', num2str(SNR_dB), ' dB)']);
legend('MUSIC', 'MUSICToeplitz');
grid on; 
