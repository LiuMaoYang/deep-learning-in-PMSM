function model = DLnet(hidden_dims, input_dim, output_dim, opt)
% Initialize a new FullyConnectedNet.
%
% Inputs:
% - hidden_dims: A list of integers giving the size of each hidden layer.
% - input_dim: An integer giving the size of the input.
% - output_dim: An integer giving the size of the output.
% - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
% the network should not use dropout at all.
% - use_batchnorm: Whether or not the network should use batch normalization.
% - reg: Scalar giving L2 regularization strength.
% - weight_scale: Scalar giving the standard deviation for random
% initialization of the weights.
weight_scale = opt.weight_scale;

model.use_batchnorm = opt.use_batchnorm;
model.use_dropout = opt.use_dropout > 0;
model.reg = opt.reg;
model.num_layers = 1 + length(hidden_dims);
model.params = {};
model.active = opt.active;
model.optim = opt.optim;

if model.use_dropout
    model.dropout_param = struct('mode', 'train', 'p', opt.use_dropout);
end
if model.use_batchnorm
    gamma = cell(0,1);
    beta = cell(0,1);
    model.bn_params = {};
end

% Initialize the parameters of the network, storing all values in
% the model.params dictionary. Store weights and biases for the first layer
% in W1 and b1; for the second layer use W2 and b2, etc.
% Weights should be initialized from a normal distribution with standard deviation
% equal to weight_scale and biases should be initialized to zero.
%
% When using batch normalization, store scale and shift parameters for the
% first layer in gamma1 and beta1; for the second layer use gamma2 and
% beta2, etc.
% Scale parameters should be initialized to one and shift parameters should be initialized to zero.

W = cell(0,1);
b = cell(0,1);

layer_input_dim = input_dim;
for i=1:length(hidden_dims)
    weight_scale = 1 / sqrt(hidden_dims(i));
    W = [W; weight_scale * randn(layer_input_dim, hidden_dims(i))];
    b = [b; weight_scale * zeros(1,hidden_dims(i))];
    if model.use_batchnorm
        gamma = [gamma; ones(1,hidden_dims(i))];
        beta = [beta; zeros(1,hidden_dims(i))];
    end
    if model.use_batchnorm
        model.bn_params = [model.bn_params; struct('mode', 'train', 'eps', opt.eps, 'momentum', opt.momentum)];
    end
    layer_input_dim = hidden_dims(i);
end
weight_scale = 1 / sqrt(output_dim);
W = [W; weight_scale * randn(layer_input_dim, output_dim)];
b = [b; weight_scale * zeros(1,output_dim)];

params.W = W;
params.b = b;
if model.use_batchnorm
    params.gamma = gamma;
    params.beta = beta;
end
model.params = params;

end
