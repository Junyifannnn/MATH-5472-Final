function [X, error] = baseline(para)
    N_iter = para.N_iter;
    W = para.weight;
    M = para.matrix;
    flag = para.flag_relaxation;
    lambda = para.lambda;
    k = para.k;
    tol = para.epsilon;
    
    X = zeros(size(M));
    loss = zeros(N_iter);
    error = zeros(N_iter);
    N_end = N_iter;
    
    for i = 1:N_iter
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
        
        [U,S,V] = svd(tmp, 'econ');
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