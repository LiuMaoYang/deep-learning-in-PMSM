function [next_x, config] = optimm(x, dx, opt, config)
if strcmpi(opt, 'sgd')
    [next_x, config] = sgd(x, dx, config);
elseif strcmpi(opt, 'adam')
    [next_x, config] = adam(x, dx, config);
end
end

function [next_x, config] = sgd(x, dx, config)
%     Performs vanilla stochastic gradient descent.
%     - learning_rate: Scalar learning rate.
%     x[t+1] = x[t] - lr * f'(x[t])
next_x = x - config.learning_rate * dx;
end

function [next_x, config] = adam(x, dx, config)
%     Uses the Adam update rule, which incorporates moving averages of both the
%     gradient and its square and a bias correction term.
%
%     config format:
%     - learning_rate: Scalar learning rate.
%     - beta1: Decay rate for moving average of first moment of gradient.
%     - beta2: Decay rate for moving average of second moment of gradient.
%     - epsilon: Small scalar used for smoothing to avoid dividing by zero.
%     - m: Moving average of gradient.
%     - v: Moving average of squared gradient.
%     - t: Iteration number.
%
%     m = beta1*m + (1-beta1)*dx
%     v = beta2*v + (1-beta2)*(dx**2)
%     x += - learning_rate * m / (np.sqrt(v) + eps)
config.t = config.t + 1;
config.m = config.beta1 * config.m +...
    (1 - config.beta1) * dx;
config.v = config.beta2 * config.v + ...
    (1 - config.beta2) * (dx .^ 2);
mb = config.m ./ (1 - config.beta1 ^ config.t);
vb = config.v ./ (1 - config.beta2 ^ config.t);
next_x = x - config.learning_rate * mb ./ (sqrt(vb) + config.epsilon);
end