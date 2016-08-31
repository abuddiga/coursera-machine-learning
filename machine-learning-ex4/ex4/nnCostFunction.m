function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% fprintf('Theta1_grad size:\n  ');
% fprintf('%d ', size(Theta1_grad));
% fprintf('Theta2_grad size:\n  ');
% fprintf('%d ', size(Theta2_grad));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Adding column of ones, bias unit (a_0^(1))
X = [ones(m,1), X];

% Calculate cost function
for i = 1:m
    a_1 = X(i, :)';
    z_2 = Theta1 * a_1;
    % include bias unit (a_0^(2))
    a_2 = [1; sigmoid(z_2)];
    z_3 = Theta2 * a_2;
    h = sigmoid(z_3);
    
    % Convert y to a binary output matrix
    newY = zeros(1, num_labels);
    newY(1, y(i)) = 1;
    
    % Calculate delta errors
    actual = (1:num_labels == y(i))'; % logical array
    delta_3 = h - actual;
    delta_2 = Theta2' * delta_3 .* [1; sigmoidGradient(z_2)];
    
    % Remove delta from bias element in 2nd layer (d_0^(2))
    delta_2 = delta_2(2:end);
    
    % Calculate theta gradients
    Theta2_grad = Theta2_grad + (delta_3 * a_2');
    Theta1_grad = Theta1_grad + (delta_2 * a_1');
    
    J = J + ( 1/m * sum( (-newY * log(h)) - ((1 - newY) * log(1 - h)) ) );
end

% Remove bias terms from Theta
Theta1_wo_bias = Theta1(:,2:end);
Theta2_wo_bias = Theta2(:,2:end);

% Calculate regularized cost function

reg = (lambda ./ (2 * m)) * (sum(sum(Theta1_wo_bias.^2)) + sum(sum(Theta2_wo_bias.^2)));

J = J + reg;

% % secondary cost variable
% B = 0;
% 
% % Calculate gradient
% for i = 1:m
%     a_1 = X(i, :)';
%     z_2 = Theta1 * a_1;
%     % include bias unit (a_0^(2))
%     a_2 = [1; sigmoid(z_2)];
%     z_3 = Theta2 * a_2;
%     h = sigmoid(z_3);
%     
%     % Convert y to a binary output matrix
%     newY = zeros(1, num_labels);
%     newY(1, y(i)) = 1;
%     
%     % Calculate delta errors
%     actual = (1:num_labels == y(i))'; % logical array
%     delta_3 = h - actual;
%     delta_2 = Theta2' * delta_3 .* [1; sigmoidGradient(z_2)];
%     
%     % Remove delta from bias element in 2nd layer (d_0^(2))
%     delta_2 = delta_2(2:end);
%     
%     % Calculate theta gradients
%     Theta2_grad = Theta2_grad + (delta_3 * a_2');
%     Theta1_grad = Theta1_grad + (delta_2 * a_1');
%     
%     B = B + ( 1/m * sum( (-newY * log(h)) - ((1 - newY) * log(1 - h)) ) );
% end


% Regularize theta gradients

Theta1_grad_reg = (lambda / m) .* Theta1;
Theta1_grad_reg(:,1) = 0; % no regularization needed for first column

Theta2_grad_reg = (lambda / m) .* Theta2;
Theta2_grad_reg(:,1) = 0; % no regularization needed for first column


% Calculate final theta gradients by dividing by m
% -------------------------------------------------------------

Theta1_grad = Theta1_grad / m + Theta1_grad_reg;
Theta2_grad = Theta2_grad / m + Theta2_grad_reg;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
