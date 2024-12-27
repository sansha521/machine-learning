% HW 3 Problem 2

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

Y(Y == 0) = -1;

%% Split the data into training and testing sets

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
    nsv_list = [];
    alpha_list = [];
    err_list = [];
    % p1 value
    p1 = 3;
    
    % p1 acts as the polynomial order here as seen in Steve Gunn's
    % svkernel.m
    % Experiment the value of C
    num_steps = 15;
    max_C = 7;
    for i=1:num_steps
        if i <= max_C
            c = 0.1^(max_C-i);
        else
            c = c+1;
        end

        % Train SVM
        [nsv, alpha, bias] = svc(X_train, Y_train, kernel, c, p1);
        
        % Compute SVM output
        Y_predicted = svcoutput(X_train, Y_train, X_test, kernel, alpha, bias, p1);

        % Compute SVM testing error
        err = svcerror(X_train, Y_train, X_test, Y_test, kernel, alpha, bias, p1);
        
        c_list = [c_list c];
        nsv_list = [nsv_list nsv];
        alpha_list = [alpha_list sum(alpha)];
        err_list = [err_list err];
    end

    % Plot Error vs. Degree of Polynomial for classifying data with polynomials
    figure('Name','Polyn Kernel')
    subplot(2,2,1)
    plot(1:num_steps,nsv_list)
    xticks(1:num_steps)
    xticklabels(c_list)
    ylabel('# of support vectors')
    title('NSV vs. C')
    subplot(2,2,2)
    plot(1:num_steps, alpha_list)
    xticks(1:num_steps)
    xticklabels(c_list)
    ylabel('Sum of alphas')
    title('Sum of Alphas vs. C')
    subplot(2,2,3)
    plot(1:num_steps, err_list)
    xticks(1:num_steps)
    xticklabels(c_list)
    ylabel('Error')
    title('Error vs. C')

%% Classify the data using RBFs (radial basis function) kernels
elseif strcmp('rbf', kernel)
    % record the performances changes
    c_list = [];
    nsv_list = [];
    alpha_list = [];
    err_list = [];
    % p1 value
    p1 = 4;

    % p1 stands for sigma when using rbf kernel. 
    % Experiment the value of C
    num_steps = 15;
    max_C = 7;
    for i=1:num_steps
        if i <= max_C
            c = 0.1^(max_C-i);
        else
            c = c+1;
        end

        % Train SVM
        [nsv, alpha, bias] = svc(X_train, Y_train, kernel, c, p1);
        
        % Compute SVM output
        Y_predicted = svcoutput(X_train, Y_train, X_test, kernel, alpha, bias, p1);

        % Compute SVM testing error
        err = svcerror(X_train, Y_train, X_test, Y_test, kernel, alpha, bias, p1);
        
        c_list = [c_list c];
        nsv_list = [nsv_list nsv];
        alpha_list = [alpha_list sum(alpha)];
        err_list = [err_list err];
    end

    % Plot Error vs. Degree of Polynomial for classifying data with polynomials
    figure('Name','RBF Kernel')
    subplot(2,2,1)
    plot(1:num_steps,nsv_list)
    xticks(1:num_steps)
    xticklabels(c_list)
    ylabel('# of support vectors')
    title('NSV vs. C')
    subplot(2,2,2)
    plot(1:num_steps, alpha_list)
    xticks(1:num_steps)
    xticklabels(c_list)
    ylabel('Sum of alphas')
    title('Sum of Alphas vs. C')
    subplot(2,2,3)
    plot(1:num_steps, err_list)
    xticks(1:num_steps)
    xticklabels(c_list)
    ylabel('Error')
    title('Error vs. C')
end