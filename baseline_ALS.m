function [X, error] = baseline_ALS(para)

    N_iter = para.N_iter;
    W = para.weight;
    M = para.matrix;
    flag = para.flag_relaxation;
    lambda = para.lambda;
    k = para.k;
    tol = para.epsilon;
    
    sz = size(M);
    A = rand(sz(1),k);
    B = rand(sz(2),k);
    loss = zeros(N_iter);
    error = zeros(N_iter);
    N_end = N_iter;
    
    for i = 1:N_iter
        X = A * B';
        Y = W .* M + (1-W) .* X;
        if flag == 0
            B = Y'* A /(A'*A);
        else
            B = Y'* A /(A'*A + lambda*eye(k));
        end
        X = A * B';
        Y = W .* M + (1-W) .* X;
        if flag == 0
            A = Y * B / (B'*B);
        else
            A = Y * B / (B'*B + lambda*eye(k));
        end


               
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
    end
    error = error(1:N_end);
end