function dx = dropout_backward(dout, cache)

%     Perform the backward pass for (inverted) dropout.
%
%     Inputs:
%     - dout: Upstream derivatives, of any shape
%     - cache: cell {p; mode; mask} from dropout_forward.

mode = cache{2}; mask = cache{3};

if strcmpi(mode, 'train')  
    dx = dout .* mask;   
elseif strcmpi(mode, 'test')
    dx = dout;
end

end