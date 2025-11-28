function R_toeplitz = toeplitz_prior(R_hat)
    M = size(R_hat, 1);%返回R_hat的行数
    R_toeplitz = zeros(M, M);
    for m = -M+1 : M-1
       
        J_m = diag(ones(M - abs(m), 1), m);
        J_mt = J_m.';
        trace_value = trace(R_hat * J_m);
        weight = 1 / (M - abs(m));
        R_toeplitz =  R_toeplitz + weight * trace_value * J_mt;
    end
end