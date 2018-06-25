function [out, cache] = dropout_forward(x, dropout_param)
%     Performs the forward pass for (inverted) dropout.
%     Inputs:
%     - x: Input data, of any shape
%       - p: Dropout parameter. We drop each neuron output with probability p.
%       - mode: 'test' or 'train'. If the mode is train, then perform dropout;
%         if the mode is test, then just return the input.
%     Outputs:
%     - out: Array of the same shape as x.
%     - cache: cell {p; mode; mask}. In training mode, mask is the dropout
%       mask that was used to multiply the input; in test mode, mask is None.
p = dropout_param.p;
mode = dropout_param.mode;
mask = zeros(0,1);
cache = {};
if strcmpi(mode, 'train')
    mask = (rand(size(x)) >= p) / (1 - p);
    out = x .* mask;
    cache = {p; mode; mask};
elseif strcmpi(mode, 'test')
    out = x;
end
end