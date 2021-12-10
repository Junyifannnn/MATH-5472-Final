clear
clc
%% Parameters setting %%
n = 1000;
p = 100;
r = 70;
sigma = 1;
epsilon = 1e-8;
N_iter = 300;
use_relaxation = 0;
lambda = 5; % For soft threshold (use_relaxation = 1)/ In paper: 5, 30, 100
k = 50; % For hard threshold (use_relaxation = 0)/ In paper: 20, 50, 70
m = 3; % Number of columns in residual matrix in Anderson algorithm

A = randn(n,r);
B = randn(p,r);
E = sigma * randn(n,p);
W = rand(n,p);
M = A * B' + E; % observed matrix

para = struct('N_iter',N_iter,'weight',W,'matrix',M,'k',k,'lambda',lambda,...
    'epsilon',epsilon,'flag_relaxation',use_relaxation,'anderson_num',m);

%% Baseline, Nesterov, Anderson %%

% ----- reproduce Figure 1 of the paper ----- %

% [X_baseline, error_baseline] = baseline(para);
% [X_nestrov, error_nesterov] = Nestrov(para);
% [X_anderson, error_anderson] = Anderson(para);
% 
% figure; hold on;
% plot(error_baseline)
% plot(error_nesterov)
% plot(error_anderson)
% % hold off
% 
% legend('baseline','nesterov','anderson')
% 
% set(gca,'FontSize', 18);
% grid on
% grid minor
% xlim([0 100]);
% set(gca,'XTickLabel',0:25:125);
% ylim([-8 -1]);
% xlabel('Iteration');
% ylabel('$\log(\Delta)$','interpreter','latex', 'FontWeight','bold');

%% Anderson with regularization %%
% [X_randerson_0, error_randerson_0, alpha_0] = Anderson_regularization(para, 0);
% [X_randerson_0_1, error_randerson_0_1, alpha_0_1] = Anderson_regularization(para, 0.1);
% [X_randerson_1, error_randerson_1, alpha_1] = Anderson_regularization(para, 1);
% [X_randerson_10, error_randerson_10, alpha_10] = Anderson_regularization(para, 10);

% ----- reproduce Figure 2 of the paper ----- %
% figure(1)
% plot(alpha_0')
% xlim([0 100]);
% hold on
% figure(2)
% plot(alpha_0_1')
% xlim([0 100]);
% hold on 
% figure(3)
% plot(alpha_1')
% xlim([0 100]);
% hold on
% figure(4)
% plot(alpha_10')
% xlim([0 100]);
% hold on

% ----- reproduce Figure 3 of the paper ----- %
% figure; hold on;
% plot(error_randerson_0,'b')
% plot(error_randerson_0_1,'r')
% plot(error_randerson_1,'k')
% plot(error_randerson_10,'m')
% hold off
% legend('0','0.1','1','10')
% set(gca,'FontSize', 18);
% grid on
% grid minor
% xlim([0 100]);
% set(gca,'XTickLabel',0:25:100);
% ylim([-8 0]);
% xlabel('Iteration');
% ylabel('$\log(\Delta)$','interpreter','latex', 'FontWeight','bold');

%% ALS relevant methods %%
% [X_baseline_ALS, error_baseline_ALS, ~, ~] = baseline_ALS(para);
% [X_baseline_ALS_sparse, error_baseline_ALS_sparse] = baseline_ALS_sparse(para);
% [X_ALS_Nesterov, error_ALS_Nesterov, Time_ALS_Nesterov] = ALS_Nesterov(para);
% [X_ALS_Anderson, error_ALS_Anderson, Time_ALS_Anderson] = ALS_Anderson(para);

%% MovieLens Data and relevant simulations %%
data = importdata('ratings.dat');
user = data(:, 1);
movies = data(:, 3);
rating = data(:, 5);
M_ml = sparse(user, movies, rating);
W_ml = M_ml ~= 0;
sz = size(M_ml);

N_iter = 400; % 400
lambda_ml = 40; % In paper 40, 50, 100
k = 100;
epsilon_ml = 1e-8;
m_ml = 3;
use_relaxation = 1;

A0 = randn(sz(1), k);
B0 = randn(sz(2), k);
para_init = struct('N_iter',100,'weight',ones(sz),'matrix',M_ml,'k',k,'lambda',lambda_ml,...
    'epsilon',epsilon_ml,'flag_relaxation',1,'anderson_num',m_ml,'A_init',A0,...
    'B_init',B0);

[A0, B0, ~, ~] = baseline_ALS_sparse(para_init);

para_ALS = struct('N_iter',N_iter,'weight',W_ml,'matrix',M_ml,'k',k,'lambda',lambda_ml,...
    'epsilon',epsilon_ml,'flag_relaxation',use_relaxation,'anderson_num',m_ml,'A_init',A0,...
    'B_init',B0);

[A_ALS_Baseline, B_ALS_Baseline, error_ALS_Baseline, T_Baseline, r_baseline] = baseline_ALS_sparse(para_ALS);
[A_ALS_Nesterov, B_ALS_Nesterov, error_ALS_Nesterov, T_Nesterov, r_Nesterov] = ALS_Nesterov(para_ALS);
[A_ALS_Anderson, B_ALS_Anderson, error_ALS_Anderson, T_Anderson, r_Anderson] = ALS_Anderson(para_ALS);

figure(1); hold on;
plot(T_Baseline, error_ALS_Baseline)
plot(T_Nesterov, error_ALS_Nesterov)
plot(T_Anderson, error_ALS_Anderson)
hold off

legend('baseline','nesterov','anderson')

ylim([-8 0]);
xlabel('Iteration');
ylabel('$\log_{\Delta}$','interpreter','latex', 'FontWeight','bold');

figure(2); hold on;
plot(T_Baseline, r_baseline)
plot(T_Nesterov, r_Nesterov)
plot(T_Anderson, r_Anderson)
hold off

legend('baseline','nesterov','anderson')




