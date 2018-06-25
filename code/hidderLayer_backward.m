function varargout = hidderLayer_backward(dout, cache, opt)
if strcmpi(opt, 'bn_relu')
    [dx, dw, db, dgamma, dbeta] = affine_bn_relu_backward(dout, cache);
    varargout{1} = dx; varargout{2} = dw; varargout{3} = db;
    varargout{4} = dgamma; varargout{5} = dbeta;
elseif strcmpi(opt, 'bn_sig')
    [dx, dw, db, dgamma, dbeta] = affine_bn_sig_backward(dout, cache);
    varargout{1} = dx; varargout{2} = dw; varargout{3} = db;
    varargout{4} = dgamma; varargout{5} = dbeta;
elseif strcmpi(opt, 'bn_tanh')
    [dx, dw, db, dgamma, dbeta] = affine_bn_tanh_backward(dout, cache);
    varargout{1} = dx; varargout{2} = dw; varargout{3} = db;
    varargout{4} = dgamma; varargout{5} = dbeta;
elseif strcmpi(opt, 'bn')  
    [dx, dgamma, dbeta] = batchnorm_backward(dout, cache);
    varargout{1} = dx; varargout{2} = dgamma; varargout{3} = dbeta;
elseif strcmpi(opt, 'relu')
%     dx = relu_backward(dout, cache);
%     varargout{1} = dx;
    [dx, dw, db] = affine_relu_backward(dout, cache);
    varargout{1} = dx; varargout{2} = dw; varargout{3} = db;
elseif strcmpi(opt, 'tanh')
    [dx, dw, db] = affine_tanh_backward(dout, cache);
    varargout{1} = dx; varargout{2} = dw; varargout{3} = db;
elseif strcmpi(opt, 'sig')
    [dx, dw, db] = affine_sig_backward(dout, cache);
    varargout{1} = dx; varargout{2} = dw; varargout{3} = db;
else
    [dx, dw, db] = affine_backward(dout, cache);
    varargout{1} = dx; varargout{2} = dw; varargout{3} = db;
end
end

function [dx, dw, db, dgamma, dbeta] = affine_bn_relu_backward(dout, cache)
[fc_cache, bn_cache, re_cache] = cache{:};
dre = relu_backward(dout, re_cache);
[dbn, dgamma, dbeta] = batchnorm_backward(dre, bn_cache);
[dx, dw, db] = affine_backward(dbn, fc_cache);
end

function [dx, dw, db, dgamma, dbeta] = affine_bn_sig_backward(dout, cache)
[fc_cache, bn_cache, sig_cache] = cache{:};
dsg = sigmoid_backward(dout, sig_cache);
[dbn, dgamma, dbeta] = batchnorm_backward(dsg, bn_cache);
[dx, dw, db] = affine_backward(dbn, fc_cache);
end

function [dx, dw, db, dgamma, dbeta] = affine_bn_tanh_backward(dout, cache)
[fc_cache, bn_cache, tanh_cache] = cache{:};
dta = sigmoid_backward(dout, tanh_cache);
[dbn, dgamma, dbeta] = batchnorm_backward(dta, bn_cache);
[dx, dw, db] = affine_backward(dbn, fc_cache);
end

function [dx, dw, db] = affine_relu_backward(dout, cache)
%     Backward pass for the affine-relu convenience layer
fc_cache = cache{1}; relu_cache = cache{2};
da = relu_backward(dout, relu_cache);
[dx, dw, db] = affine_backward(da, fc_cache);
end

function [dx, dw, db] = affine_tanh_backward(dout, cache)
fc_cache = cache{1}; tanh_cache = cache{2};
da = tanh_backward(dout, tanh_cache);
[dx, dw, db] = affine_backward(da, fc_cache);
end

function [dx, dw, db] = affine_sig_backward(dout, cache)
%     Backward pass for the affine-relu convenience layer
fc_cache = cache{1}; sig_cache = cache{2};
da = sigmoid_backward(dout, sig_cache);
[dx, dw, db] = affine_backward(da, fc_cache);
end

function [dx, dw, db] = affine_backward(dout, cache)
%     Computes the backward pass for an affine layer.
%
%     Inputs:
%     - dout: Upstream derivative, of shape (N, M)
%     - cache: Tuple of:
%       - x: Input data, of shape (N, d_1, ... d_k)
%       - w: Weights, of shape (D, M)
%
%     out = x * w + b
%     dout: dloss/dout
%     dx: dloss/dx   dx = dloss/dout * dout/dx = dout * wT (N, M) * (M, D) -->(N, D) -->(N, d1,...dk)
%     dw: dloss/dw   dw = xT * dout (D, N) * (N, M)--->(D, M)
%     db: dloss/db   db = dout.T * ones(N, )  (M, N) * (N, ) --->(M, )
%
%     Returns a tuple of:
%     - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
%     - dw: Gradient with respect to w, of shape (D, M)
%     - db: Gradient with respect to b, of shape (M,)

x = cache{1}; w = cache{2};
% [N,~] = size(x);

dx = dout * w';
dw = x' * dout;
db = sum(dout);
end

function dx = sigmoid_backward(dout, cache)
x = cache;
dx = dout .* (1 - dout);
end

function dx = tanh_backward(dout, cache)
x = cache;
dx = 1 - dout .^2;
end

function dx = relu_backward(dout, cache)
%     Computes the backward pass for a layer of rectified linear units (ReLUs).
%
%     Input:
%     - dout: Upstream derivatives, of any shape
%     - cache: Input x, of same shape as dout
%
%     Returns:
%     - dx: Gradient with respect to x
%
%     out = max(0, x)
%     The max gate routes the gradient.
%     Unlike the add gate which distributed the gradient unchanged to all its inputs,
%     the max gate distributes the gradient (unchanged) to exactly one of its inputs
%     (the input that had the highest value during the forward pass).
x = cache;
dx = dout .* (x >= 0) + 0.01 * dout .* (x < 0);
end

function [dx, dgamma, dbeta] = batchnorm_backward(dout, cache)
%     Backward pass for batch normalization.
%
%     For this implementation, you should write out a computation graph for
%     batch normalization on paper and propagate gradients backward through
%     intermediate nodes.
%
%     Inputs:
%     - dout: Upstream derivatives, of shape (N, D)
%     - cache: Variable of intermediates from batchnorm_forward.
%
%     Returns a tuple of:
%     - dx: Gradient with respect to inputs x, of shape (N, D)
%     - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
%     - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
%
%     mean = 1 / m * sum(x)
%     var = 1 / m * sum((x - mean)^2)
%     x_norm = (x - mean) / sqrt(var + eps)
%     out = gamma * x_norm + beta
%
%     dout = dloss / dout
%     dx_norm = gamma * dout
%
%     dx_norm / dvar = (-1/2) * (x - mean) * (var + eps)^(-3/2)
%     dvar = dx_norm * dx_norm / dvar
%
%     dvar / dmean = (-2 / m) * sum(x - mean)
%     dx_norm / dmean = -1 / sqrt(var + eps) + dx_norm / dvar * dvar / dmean
%     dmean / dx = 1
%     dvar / dx = (2 / m) * sum(x - mean) * dmean / dx
%     dx_norm / dx = (1 - dmean / dx) * (1 / sqrt(var + eps)) + dx_norm / dvar * dvar / dmean
%
%     dx = dx_norm *

%     mean = 1 / m * sum(x)
%     var = 1 / m * sum((x - mean)^2)
%     x_norm = (x - mean) / sqrt(var + eps)
%     out = gamma * x_norm + beta

[gamma, x, mean, var, eps, x_hat, ~] = cache{:};
N = size(x, 1);
dx_hat = dout .* gamma;

dgamma = sum(dout .* x_hat);
dbeta = sum(dout);

dvar = - 0.5 * sum(dx_hat .* (x - mean) .* power(var + eps, -1.5));
dmean = sum(-1 * dx_hat ./ sqrt(var + eps)) + dvar .* sum(-2 * (x - mean)) ./N;

dx = dx_hat ./ sqrt(var + eps) + 2 / N * dvar .* (x - mean) + dmean / N;
end


