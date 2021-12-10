function [A, B, error, T, rank_X] = baseline_ALS_sparse(para)
    N_iter = para.N_iter;
    W = para.weight;
    M = para.matrix;
    flag = para.flag_relaxation;
    lambda = para.lambda;
    k = para.k;
    tol = para.epsilon;
    
    sz = size(M);
    A = para.A_init;
    B = para.B_init;
    loss = zeros(N_iter);
    error = zeros(N_iter);
    T = zeros(N_iter);
    rank_X = zeros(N_iter);
    N_end = N_iter;
    
    tic
    for i = 1:N_iter
        S = W .* (M - A * B');
        if flag == 0
            H_A = A / (A'*A);
        else
            H_A =  A /(A'*A + lambda*eye(k));
        end
        B = B * A' * H_A + S' * H_A;
        S = W .* (M - A * B');
        if flag == 0
            H_B = B / (B'*B);
        else
            H_B = B / (B'*B + lambda*eye(k));
        end
        A = A * B' * H_B + S * H_B;
               
        if flag == 0
            loss(i) = sum(W.*((M-A*B').^2), 'all');
        else
            loss(i) = 0.5 * sum(W.*((M-A*B').^2), 'all') + 0.5 * lambda * (sum(A.^2, 'all') + sum(B.^2, 'all'));
        end
        rank_X(i) = find_rank(A, B);
        
        if i > 1
            error(i-1) = log10(abs((loss(i) - loss(i-1)))) - log10(abs(loss(i-1)));
            disp(['Iteration: ',num2str(i), ' Error: ',num2str(error(i-1))])
            if abs((loss(i) - loss(i-1))/loss(i-1)) < tol
                N_end = i;
                break
            end
        end
        T(i) = toc;
    end
    rank_X = rank_X(1:N_end-1);
    error = error(1:N_end-1);
    T = T(1:N_end-1);
end