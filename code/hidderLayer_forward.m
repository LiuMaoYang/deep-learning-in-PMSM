function [out, cache] = hidderLayer_forward(x, w, b, opt, varargin)
if strcmpi(opt, 'bn_relu')
    gamma = varargin{1};
    beta = varargin{2};
    bn_param = varargin{3};
    [out, cache] = affine_bn_relu_forward(x, w, b, gamma, beta, bn_param);
elseif strcmpi(opt, 'bn_sig')
    gamma = varargin{1};
    beta = varargin{2};
    bn_param = varargin{3};
    [out, cache] = affine_bn_sig_forward(x, w, b, gamma, beta, bn_param);
elseif strcmpi(opt, 'bn_tanh')
    gamma = varargin{1};
    beta = varargin{2};
    bn_param = varargin{3};
    [out, cache] = affine_bn_tanh_forward(x, w, b, gamma, beta, bn_param);
elseif strcmpi(opt, 'bn')
    gamma = varargin{1};
    beta = varargin{2};
    bn_param = varargin{3};
    [out, cache] = batchnorm_forward(x, gamma, beta, bn_param);
elseif strcmpi(opt, 'relu')
%     [out, cache] = relu_forward(x);
    [out, cache] = affine_relu_forward(x, w, b);
elseif strcmpi(opt, 'tanh')
    [out, cache] = affine_tanh_forward(x, w, b);
elseif strcmpi(opt, 'sig')
    [out, cache] = affine_sig_forward(x, w, b);
else
    [out, cache] = affine_forward(x, w, b);
end
end

function [out, cache] = affine_bn_relu_forward(x, w, b, gamma, beta, bn_param)
[fc_out, fc_cache] = affine_forward(x, w, b);
[bn_out, bn_cache] = batchnorm_forward(fc_out, gamma, beta, bn_param);
[out, re_cache] = relu_forward(bn_out);
cache = {fc_cache; bn_cache; re_cache};
end

function [out, cache] = affine_bn_sig_forward(x, w, b, gamma, beta, bn_param)
[fc_out, fc_cache] = affine_forward(x, w, b);
[bn_out, bn_cache] = batchnorm_forward(fc_out, gamma, beta, bn_param);
[out, sig_cache] = sigmoid_forward(bn_out);
cache = {fc_cache; bn_cache; sig_cache};
end

function [out, cache] = affine_bn_tanh_forward(x, w, b, gamma, beta, bn_param)
[fc_out, fc_cache] = affine_forward(x, w, b);
[bn_out, bn_cache] = batchnorm_forward(fc_out, gamma, beta, bn_param);
[out, tanh_cache] = tanh_forward(bn_out);
cache = {fc_cache; bn_cache; tanh_cache};
end

function [out, cache] = affine_relu_forward(x, w, b)
[a, fc_cache] = affine_forward(x, w, b);
[out, relu_cache] = relu_forward(a);
cache = {fc_cache; relu_cache};
end

function [out, cache] = affine_tanh_forward(x, w, b)
[a, fc_cache] = affine_forward(x, w, b);
[out, tanh_cache] = tanh_forward(a);
cache = {fc_cache; tanh_cache};
end

function [out, cache] = affine_sig_forward(x, w, b)
[a, fc_cache] = affine_forward(x, w, b);
[out, sig_cache] = sigmoid_forward(a);
cache = {fc_cache; sig_cache};
end

function [out, cache] = affine_forward(x, w, b)
% Computes the forward pass for an affine (fully-connected) layer.
%
% The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
%     examples, where each example x[i] has shape (d_1, ..., d_k). We will
%     reshape each input into a vector of dimension D = d_1 * ... * d_k, and
%     then transform it to an output vector of dimension M.
%
% Inputs:
% - x: A array containing input data, of shape (N, d_1, ..., d_k)
%     - w: A array of weights, of shape (D, M) D = d1 * ...* dk
%     - b: A array of biases, of shape (M,)
%
% out = x * w + b
%
% Returns:
% - out: output, of shape (N, M)
% - cache: {x; w; b}

out = x * w + b ;
cache = {x; w; b};
end

function [out, cache] = tanh_forward(x)
a = exp(x);
b = exp(-x);
out = (a - b) ./ (a + b);
cache = x;
end

function [out, cache] = sigmoid_forward(x)
out = 1 ./ (1 + exp(-x));
cache = x;
end

function [out, cache] = relu_forward(x)
%     Computes the forward pass for a layer of rectified linear units (ReLUs).
%
%     Input:
%     - x: Inputs, of any shape
%
%     Returns:
%     - out: Output, of the same shape as x
%     - cache: x
%
%     out = max(0, x)

out = x .* (x >= 0);
out(out == 0) = 0.01;
cache = x;
end

function [out, cache] = batchnorm_forward(x, gamma, beta, bn_param)
%     During training the sample mean and (uncorrected) sample variance are
%     computed from minibatch statistics and used to normalize the incoming data.
%     During training we also keep an exponentially decaying running mean of the
%     mean and variance of each feature, and these averages are used to normalize
%     data at test-time.
%
%     At each timestep we update the running averages for mean and variance using
%     an exponential decay based on the momentum parameter:
%
%     running_mean = momentum * running_mean + (1 - momentum) * sample_mean
%     running_var = momentum * running_var + (1 - momentum) * sample_var
%
%     Note that the batch normalization paper suggests a different test-time
%     behavior: they compute sample mean and variance for each feature using a
%     large number of training images rather than using a running average. For
%     this implementation we have chosen to use running averages instead since
%     they do not require an additional estimation step; the torch7
%     implementation of batch normalization also uses running averages.
%
%     Input:
%     - x: Data of shape (N, D)
%     - gamma: Scale parameter of shape (D,)
%     - beta: Shift paremeter of shape (D,)
%         means should be close to beta and stds close to gamma
%
%     - bn_param: Dictionary with the following keys:
%       - mode: 'train' or 'test'; required
%       - eps: Constant for numeric stability
%       - momentum: Constant for running mean / variance.
%       - running_mean: Array of shape (D,) giving running mean of features
%       - running_var Array of shape (D,) giving running variance of features
%
%     Returns a tuple of:
%     - out: of shape (N, D)
%     - cache: A tuple of values needed in the backward pass

mode = bn_param.mode;
eps = bn_param.eps;
momentum = bn_param.momentum;

D = size(x, 2);
if isfield(bn_param, 'running_mean')==0 || isfield(bn_param, 'running_var')==0
    running_mean = zeros(1, D);
    running_var = zeros(1, D);
else
    running_mean = bn_param.running_mean;
    running_var = bn_param.running_var;
end

cache = {};
if strcmpi(mode, 'train')
    sample_mean = mean(x);% (D, )
%     sample_var = var(x);
    sample_var = mean(abs(x - mean(x)).^2);
    x_norm = (x - sample_mean) ./ sqrt(sample_var + eps);% normalize
    out = gamma .* x_norm + beta; % scale and shift for each col
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean;
    running_var = momentum * running_var + (1 - momentum) * sample_var;
    
    %Store the updated running means back into bn_param
    bn_param.running_mean = running_mean;
    bn_param.running_var= running_var;
    cache = {gamma; x; sample_mean; sample_var; eps; x_norm; bn_param};
    
elseif strcmpi(mode, 'test')
    scale = gamma ./ (sqrt(running_var + eps));
    out = x .* scale + (beta - running_mean .* scale); 
end

end



