function R_shrink = linear_shrinkage(R_hat, N)
    R_T = toeplitz_prior(R_hat); % 托普利兹目标矩阵

    % 计算收缩系数 alpha
    numerator = (N-3) * trace(R_hat * R_hat) + (N-1) * trace(R_hat)^2;
    denominator = (N-2) * (N+1) * trace((R_hat - R_T)*(R_hat - R_T));
    alpha = min(numerator / denominator, 1);

    % 线性收缩
    R_shrink = (1 - alpha) * R_hat + alpha * R_T;
end