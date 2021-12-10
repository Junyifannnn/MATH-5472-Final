function [X, error, alpha] = Anderson_regularization(para, gamma)
    N_iter = para.N_iter;
    W = para.weight;
    M = para.matrix;
    flag = para.flag_relaxation;
    lambda = para.lambda;
    k = para.k;
    tol = para.epsilon;
    m = para.anderson_num;
    N_end = N_iter;
    gamma_tmp = gamma;
    gamma = 0;
    
    size_M = size(M);
    size_vec = prod(size_M);
    loss = zeros(N_iter);
    error = zeros(N_iter);
    
%%   initialization    %%
    X = zeros(size_M);
    Y = X;
    y = Y(:);
    
    R = zeros(size_vec, m+1);
    F = zeros(size_vec, m+1);
    alpha = zeros(m+1, N_iter);
    

%% Main Loop %%
    for i = 1:N_iter
        if i > m
            gamma = gamma_tmp;
        end
        tmp = W.*M + (1-W).*X;
        
        if flag == 0
            loss(i) = norm(sqrt(W).*(M-X), 'fro')^2;
        else
            loss(i) = 0.5 * norm(sqrt(W).*(M-X), 'fro')^2 + lambda * norm(svd(X), 1);
        end
        
        if i > 1
            error(i-1) = log10(abs((loss(i) - loss(i-1)))) - log10(abs(loss(i-1)));
            if abs((loss(i) - loss(i-1))/loss(i-1)) < tol
                N_end = i;
                break
            end
        end
        
        f = tmp(:);
        r = f - y;
        if i <= m+1
            R(:,i) = r;
            F(:,i) = f;
        else
            R = [R(:,2:end) r];
            F = [F(:,2:end) f];
        end
        
        
%% regularization %%

        col_size = size(R,2);
        [alpha_star, ~] = lsqr(R'*R+gamma*eye(col_size), ones(col_size,1));
        alpha_star = alpha_star / sum(alpha_star);
        if i <= m
            alpha_prev = mean(alpha(:,1:i), 2);
        else
            alpha_prev = mean(alpha(:,i-m:i-1),2);
        end

        if gamma == 0
            K = 0;
        else
            K = (R'*R+gamma*eye(col_size)) \ (alpha_prev*ones(col_size,1)'-ones(col_size,1)*alpha_prev');
        end
        alpha_next = (eye(col_size)+gamma*K)*alpha_star;
        alpha(:,i) = alpha_next;
        
%% Thresholding update %%

        y = F * alpha_next;
        Y = reshape(y, size_M);

        [U,S,V] = svd(Y, 'econ');
        if flag == 0
            S(k+1:end, k+1:end) = 0;
            X = U*S*V';
        else
            diag_S = diag(S);
            diag_S = max(diag_S-lambda,0);
            X = U*diag(diag_S)*V';
        end
    end
    
    alpha = alpha(:,1:N_end-1);
    error = error(1:N_end-1);
end