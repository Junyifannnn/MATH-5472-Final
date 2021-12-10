function [A, B, error, T, rank_X] = ALS_Anderson(para)

    N_iter = para.N_iter;
    W = para.weight;
    M = para.matrix;
    flag = para.flag_relaxation;
    lambda = para.lambda;
    k = para.k;
    tol = para.epsilon;
    m = para.anderson_num;
    
    sz = size(M);
    
    loss = zeros(N_iter);
    error = zeros(N_iter);
    T = zeros(N_iter);
    rank_X = zeros(N_iter);
    N_end = N_iter;
    
%% Initialization %%
    A = para.A_init;
    B = para.B_init;
    size_vec_A = numel(A);
    size_vec_B = numel(B);
    
    R = zeros(size_vec_A+size_vec_B, m+1);
    F = zeros(size_vec_A+size_vec_B, m+1);

    if flag == 0
        loss(1) = sum(W.*((M-A*B').^2), 'all');        
    else
        loss(1) = 0.5 * sum(W.*((M-A*B').^2), 'all') + 0.5 * lambda * (sum(A.^2, 'all') + sum(B.^2, 'all'));
    end

    tic;
    for i = 1:N_iter
%% Main Loop of ALS %%

        S = W .* (M - A * B');
        if flag == 0
            H_A = A / (A'*A);
        else
            H_A =  A /(A'*A + lambda*eye(k));
        end
        B_next = B * A' * H_A + S' * H_A;
        S = W .* (M - A * B_next');
        if flag == 0
            H_B = B_next / (B_next'*B_next);
        else
            H_B = B_next / (B_next'*B_next + lambda*eye(k));
        end
        A_next = A * B_next' * H_B + S * H_B;
       
%% Anderson_update %%
        Y = [A; B];
        y = Y(:);
        F_tmp = [A_next; B_next];
        f = F_tmp(:);
        r = f - y;
        
        if i <= m+1
            R(:, i) = r;
            F(:, i) = f;
        else
            R = [R(:,2:end) r];
            F = [F(:,2:end) f];

        end
        
        col_size = size(R,2);
        [sita, ~] = lsqr(R'*R, ones(col_size,1));
        alpha = sita / sum(sita);
        
        y = F * alpha;
        AB = reshape(y, size(Y));
        A_Anderson = AB(1:sz(1),:);
        B_Anderson = AB(sz(1)+1:end,:);

%% Calculate Loss %%
        if flag == 0
            loss_Anderson = sum(W.*((M-A*B').^2), 'all');
            loss_ALS = sum(W.*((M-A_next*B_next').^2), 'all');
        else
            loss_Anderson = 0.5 * sum(W.*((M-A_Anderson*B_Anderson').^2), 'all') + 0.5 * lambda * (sum(A_Anderson.^2, 'all') + sum(B_Anderson.^2, 'all'));
            loss_ALS = 0.5 * sum(W.*((M-A_next*B_next').^2), 'all') + 0.5 * lambda * (sum(A_next.^2, 'all') + sum(B_next.^2, 'all'));
        end
%% Compare the loss with ALS baseline (guarded acceleration) %%        
        if loss_Anderson < loss_ALS
            loss(i+1) = loss_Anderson;
            A = A_Anderson;
            B = B_Anderson;
        else
            loss(i+1) = loss_ALS;
            A = A_next;
            B = B_next;
        end
        rank_X(i) = find_rank(A, B);
          
        error(i) = log10(abs((loss(i+1) - loss(i)))) - log10(abs(loss(i)));
        disp(['Iteration: ',num2str(i), ' Error: ',num2str(error(i))])
        if abs((loss(i+1) - loss(i))/loss(i)) < tol
            N_end = i;
            T(i) = toc;  
            break
        end
        
        T(i) = toc;

    end
    rank_X = rank_X(1:N_end);
    error = error(1:N_end);
    T = T(1:N_end);
end