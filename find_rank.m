function r = find_rank(A, B)
    [~,S_A,V_A] = svd(A, 'econ');
    B_tmp = B * V_A' * S_A;
    [~,S_B,~] = svd(B_tmp, 'econ');
    r = nnz(S_B>1e-8);
end