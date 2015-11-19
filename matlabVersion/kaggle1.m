
%% kaggle = dlmread('train.csv', ',', 1, 0);
%% X = kaggle(:, 2:end);
%% y = kaggle(:,1);
%% update = y==0
%% y(update) = 10

%% save -binary 'kaggle.mat' X y
clear;clc;
load('kaggle.mat');

%%y_temp = find(y == 0);
%%y = y(y_temp) = 10;

m = size(X, 1);

options = optimset('MaxIter', 50);
lambda = 1;
input_layer_size = 784;
hidden_layer_size = 28;
num_labels = 10;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];




costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);