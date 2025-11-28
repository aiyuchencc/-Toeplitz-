function [spectrum_scm,spectrum_toeplitz,estimated_doas_scm,estimated_doas_toeplitz] = music_test(Y,N, M, K, angle_grid)
R_hat = (Y * Y') / N;%40*40
R_shrink = linear_shrinkage(R_hat, N);

% 使用R_hat 运行MUSIC
[E1,D1] = eig(R_hat);
[~, idx1] = sort(diag(D1), 'descend');
E1 = E1(:, idx1);
En1 = E1(:, K+1:M);
spectrum_scm = zeros(length(angle_grid), 1); 
for i = 1:length(angle_grid)
     a = exp(-1j  * pi * (0:M-1)' * sind(angle_grid(i)));
    spectrum_scm(i) = 1 / (a' * (En1 * En1') * a);
end
    [~, locs] = findpeaks(abs(spectrum_scm), 'SortStr', 'descend');
    estimated_doas_scm = angle_grid(locs(1:min(K, length(locs)))); 
    estimated_doas_scm=sort(estimated_doas_scm);

% 使用R_shrink运行MUSIC
[E,D] = eig(R_shrink);
[~, idx] = sort(diag(D), 'descend');% 特征值降序排序
E = E(:, idx);%重排后的特征向量矩阵
En = E(:, K+1:M); % 提取噪声子空间（后M-K个特征向量）
spectrum_toeplitz = zeros(length(angle_grid), 1);% 构建空间谱
for i = 1:length(angle_grid)
    a = exp(-1j  * pi * (0:M-1)' * sind(angle_grid(i)));
    spectrum_toeplitz(i) = 1 / (a' * (En * En') * a);
end
[~, peaks] = findpeaks(abs(spectrum_toeplitz), 'SortStr', 'descend');%在频谱幅度中检测峰值位置，并按峰值大小降序排序。
estimated_doas_toeplitz = angle_grid(peaks(1:min(K, length(peaks))));%从排序后的峰值位置中提取前K个最强的角度估计
estimated_doas_toeplitz=sort(estimated_doas_toeplitz);
end