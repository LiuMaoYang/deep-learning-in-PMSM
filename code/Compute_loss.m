function varargout = Compute_loss(model, varargin)
% Compute loss and gradient for the fully-connected net.  varargout
if nargin==2
    mode = 'test';
    X = varargin{1};
elseif nargin==3
    mode = 'train';
    X = varargin{1};
    y = varargin{2};
end

% Normalize
% eps = 1e-5;
% sample_mean = mean(X); % (D, )
% sample_var = var(X);
% X_norm = (X - sample_mean) ./ sqrt(sample_var + eps);

if model.use_dropout
    model.dropout_param.mode = mode;
end

if model.use_batchnorm
    for i=1 : length(model.bn_params)
        model.bn_params{i}.mode = mode;
    end
end

active = model.active;

% forward step.
% layer_input = X_norm;
layer_input = X;
ar_cache = cell(model.num_layers,1);
dp_cache = cell(model.num_layers - 1,1);
for l = 1:model.num_layers - 1
    W = model.params.W{l};
    b = model.params.b{l};
    if model.use_batchnorm
        gamma = model.params.gamma{l};
        beta = model.params.beta{l};
        [layer_input, ar_cache{l}] = hidderLayer_forward(layer_input, W, b, ['bn_' active], gamma, beta, model.bn_params{l}); %affine_bn_relu_forward
        if model.use_batchnorm && strcmpi(mode, 'train')
            model.bn_params{l} = ar_cache{l}{2}{end};
        end
    else
        [layer_input, ar_cache{l}] = hidderLayer_forward(layer_input, W, b, active); %affine_relu_forward
    end
    
    if model.use_dropout
        [layer_input, dp_cache{l}] = dropout_forward(layer_input, model.dropout_param);
    end
end
W = model.params.W{model.num_layers};
b = model.params.b{model.num_layers};
[scores, ar_cache{model.num_layers}] = hidderLayer_forward(layer_input, W, b, ''); %affine_forward

% backward step.
%If test mode return early
if strcmpi(mode, 'test')
    varargout{1} = scores;
else
    grads_W = cell(model.num_layers,1);
    grads_b = cell(model.num_layers,1);
    if model.use_batchnorm
        grads_gamma = cell(model.num_layers - 1,1);
        grads_beta = cell(model.num_layers - 1,1);
    end
    
    [loss, dscores] = cost(scores, y);
    loss = loss + 0.5 * model.reg * sum(sum(model.params.W{model.num_layers}.^2));
    [dout, dw, db] = hidderLayer_backward(dscores, ar_cache{model.num_layers}, '');%affine_backward
    grads_W{model.num_layers} = dw + model.reg * model.params.W{model.num_layers};
    grads_b{model.num_layers} = db;
       
    for lr = model.num_layers - 1 : -1 : 1
        loss = loss + 0.5 * model.reg * sum(sum(model.params.W{lr}.^2));
        if model.use_dropout
            dout = dropout_backward(dout, dp_cache{lr});
        end
        
        if model.use_batchnorm
            [dout, dw, db, dgamma, dbeta] = hidderLayer_backward(dout, ar_cache{lr}, ['bn_' active]);%affine_bn_relu_backward
            grads_gamma{lr} = dgamma;
            grads_beta{lr} = dbeta;
        else
            [dout, dw, db] = hidderLayer_backward(dout, ar_cache{lr}, active);%affine_relu_backward
        end
        
        grads_W{lr} = dw + model.reg * model.params.W{lr};
        grads_b{lr} = db;
    end
    grads.W = grads_W;
    grads.b = grads_b;
    if model.use_batchnorm
        grads.gamma = grads_gamma;
        grads.beta = grads_beta;
    end
    varargout{1} = loss;
    varargout{2} = grads;
    varargout{3} = model;
end
end

% function [loss, dx] = cost(X, y)
% % loss
% % (X-y)/y
% n = length(y);
% loss = sum((X - y).^2)/n;
% dx = (X - y) ./ abs(y);
% err = X - y;
% end


function [loss, dx] = cost(X, y)
% MSE loss
% Inputs:
%     - x: Input data, of shape (N, ) where x[i, j] is the predict for the jth
%       class for the ith input.
%     - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
%       0 <= y[i] < C
% 
%     Returns a tuple of:
%     - loss: Scalar giving the loss
%     - dx: Gradient of the loss with respect to x (N, C)
n = length(y);
loss = sum((X - y).^2)/n;
dx = (X - y);
end




