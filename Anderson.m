function [X, error] = Anderson(para)

    N_iter = para.N_iter;
    W = para.weight;
    M = para.matrix;
    flag = para.flag_relaxation;
    lambda = para.lambda;
    k = para.k;
    tol = para.epsilon;
    m = para.anderson_num;
    
    size_M = size(M);
    size_vec = prod(size_M);
    loss = zeros(N_iter);
    error = zeros(N_iter);
    
%% initialization %%
    X = zeros(size_M);
    Y = X;
    y = Y(:);
    
    R = zeros(size_vec, m+1);
    F = zeros(size_vec, m+1);
   
    for i = 1:N_iter
%% Main Loop %%
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
        
%% Anderson update %%

        f = tmp(:);
        r = f - y;
        if i <= m+1
            R(:,i) = r;
            F(:,i) = f;
        else
            R = [R(:,2:end) r];
            F = [F(:,2:end) f];
        end
        
        col_size = size(R,2);
        [sita, ~] = lsqr(R'*R, ones(col_size,1));
        alpha = sita / sum(sita);

        y = F * alpha;
        Y = reshape(y, size_M);

%% SVD update %%

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
    error = error(1:N_end-1);
end