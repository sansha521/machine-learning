clear;

% set figure theme from dark to light
s = settings;
s.matlab.appearance.figure.GraphicsTheme.TemporaryValue = "light";

load('dataset.mat')

%% Choose Kernel to use in SVM
% As seen in Dr. Steve Gunn's SVM package documentation:
%  Values for ker: 'linear'  -
%                  'poly'    - p1 is degree of polynomial
%                  'rbf'     - p1 is width of rbfs (sigma)
kernel = 'poly';

%% Try normalizing data
%[X, A, B] = svdatanorm(X, kernel);
X_mean = mean(X);
X_std = std(X);
X_normalized = (X - X_mean) ./ X_std;

% NOTE: modified the data from {0, 1} to {-1, 1}
Y(Y == 0) = -1;

%% Split the data into training and testing sets
% cvpartition from the ML/stats toolbox to do data split in one line
% 0.5 for 50% for testing
cv = cvpartition(size(X, 1), 'HoldOut', 0.5);

% get indices for the training and test sets in first fold
train_idxs = training(cv);
test_idxs = test(cv);

% create training and test sets by splitting the features and labels
n = size(X, 1);
idx_train = randsample(n, n / 2);

% Split the dataset
idxtest = setdiff(1:n, idx_train);

X_train = X(idx_train, :);
X_test = X(idxtest, :);

Y_train = Y(idx_train, :);
Y_test = Y(idxtest, :);



%% Classify the data using polynomial kernels
if strcmp('poly', kernel)
    % record the performances changes
    c_list = [];
    p1_list = [];
    nsv_list = [];
    alpha_list = [];
    acc_list = [];
    % d order of the polynomial
    d_max = 10;
    % C value
    C = 1;
    
    % p1 acts as the polynomial order here as seen in Steve Gunn's
    % svkernel.m
    for i=1:d_max
        % change polynomial order
        p1 = i;

        % Train SVM
        [nsv, alpha, bias] = svc(X_train, Y_train, kernel, C, p1);
        
        % Compute SVM output
        Y_predicted = svcoutput(X_train, Y_train, X_test, kernel, alpha, bias, p1);

        % Compute SVM testing error
        %err = svcerror(X_train, Y_train, X_test, Y_test, kernel, alpha, bias, p1);
        acc = (1-sum(Y_predicted ~= Y_test) / length(Y_test));

        p1_list = [p1_list p1];
        nsv_list = [nsv_list nsv];
        alpha_list = [alpha_list sum(alpha)];
        acc_list = [acc_list acc];
        %y_pred_list = [y_pred_list Y_predicted];
    end

    % Plot Error vs. Degree of Polynomial for classifying data with polynomials
    figure('Name','Polyn Kernel')
    subplot(2,2,1)
    plot(1:d_max,nsv_list)
    xticks(1:d_max)
    xticklabels(p1_list)
    ylabel('# of support vectors')
    title('NSV vs. Polyn Order')
    subplot(2,2,2)
    plot(1:d_max, alpha_list)
    xticks(1:d_max)
    xticklabels(p1_list)
    ylabel('Sum of alphas')
    title('Sum of Alphas vs. Polyn Order')
    subplot(2,2,3)
    plot(1:d_max, acc_list)
    xticks(1:d_max)
    xticklabels(p1_list)
    ylabel('Acc')
    title('Acc vs. Polyn Order')

%% Classify the data using RBFs (radial basis function) kernels
elseif strcmp('rbf', kernel)
    % record the performances changes
    c_list = [];
    p1_list = [];
    nsv_list = [];
    alpha_list = [];
    acc_list = [];
    % d max variation of the sigma value
    d_max = 20;
    % C value
    C = 3;

    % p1 stands for sigma when using rbf kernel. 
    for i=1:d_max
        % change sigma value
        p1 = 1 + 0.5*(i - 1);

        % Train SVM
        [nsv, alpha, bias] = svc(X_train, Y_train, kernel, C, p1);
        
        % Compute SVM output
        Y_predicted = svcoutput(X_train, Y_train, X_test, kernel, alpha, bias, p1);

        % Compute SVM testing error
        %err = svcerror(X_train, Y_train, X_test, Y_test, kernel, alpha, bias, p1);
        acc = (1-sum(Y_predicted ~= Y_test) / length(Y_test));
        
        p1_list = [p1_list p1];
        nsv_list = [nsv_list nsv];
        alpha_list = [alpha_list sum(alpha)];
        acc_list = [acc_list acc];
        %y_pred_list = [y_pred_list Y_predicted];
    end

    % Plot Error vs. Degree of Polynomial for classifying data with polynomials
    figure('Name','RBF Kernel')
    subplot(2,2,1)
    plot(1:d_max,nsv_list)
    xticks(1:d_max)
    xticklabels(p1_list)
    ylabel('# of support vectors')
    title('NSV vs. Sigma')
    subplot(2,2,2)
    plot(1:d_max, alpha_list)
    xticks(1:d_max)
    xticklabels(p1_list)
    ylabel('Sum of alphas')
    title('Sum of Alphas vs. Sigma')
    subplot(2,2,3)
    plot(1:d_max, acc_list)
    xticks(1:d_max)
    xticklabels(p1_list)
    ylabel('Acc')
    title('Acc vs. Sigma')
end