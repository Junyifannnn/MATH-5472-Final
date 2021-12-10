function [A, B, error, T, rank_X] = ALS_Nesterov(para)

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
    A_next = A;
    B_next = B;
    loss = zeros(N_iter);
    error = zeros(N_iter);
    T = zeros(N_iter);
    rank_X = zeros(N_iter);
    N_end = N_iter;
    
    if flag == 0
        loss(1) = sum(W.*((M-A*B').^2), 'all');        
    else
        loss(1) = 0.5 * sum(W.*((M-A*B').^2), 'all') + 0.5 * lambda * (sum(A.^2, 'all') + sum(B.^2, 'all'));
    end

    
    tic;
    for i = 1:N_iter
%% Nesterov update %%

        V_A = A_next + (i-1) * (A_next - A) / (i+2);
        V_B = B_next + (i-1) * (B_next - B) / (i+2);
        
        A = A_next;
        B = B_next;
       
%% Main loop update of ALS %%

        S = W .* (M - V_A * V_B');
        if flag == 0
            H_A = V_A / (V_A'*V_A);
        else
            H_A =  V_A / (V_A'*V_A + lambda*eye(k));
        end
        B_next = V_B * V_A' * H_A + S' * H_A;
        S = W .* (M - V_A * B_next');
        if flag == 0
            H_B = B_next / (B_next'*B_next);
        else
            H_B = B_next / (B_next'*B_next + lambda*eye(k));
        end
        A_next = V_A * V_B' * H_B + S * H_B;
        
%% Calculate Loss %%   

        if flag == 0
            loss(i+1) = sum(W.*((M-A*B').^2), 'all');        
        else
            loss(i+1) = 0.5 * sum(W.*((M-A_next*B_next').^2), 'all') + 0.5 * lambda * (sum(A_next.^2, 'all') + sum(B_next.^2, 'all'));
        end
        rank_X(i) = find_rank(A_next, B_next);

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