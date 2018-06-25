function result = DLtrain(model, data, opt)
% Required arguments:
% - model: A model object conforming to the DLnet
% - data: A dictionary of training and validation data containing:
%     'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
%     'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
%     'y_train': Array, shape (N_train,) of labels for training images
%     'y_val': Array, shape (N_val,) of labels for validation images
%
%     Optional arguments:
%     - update_rule: A string giving the name of an update rule in optim.py.
%     Default is 'sgd'.
%     - optim_config: A dictionary containing hyperparameters that will be
%     passed to the chosen update rule. Each update rule requires different
%     hyperparameters (see optim.py) but all update rules require a
%     'learning_rate' parameter so that should always be present.
%     - lr_decay: A scalar for learning rate decay; after each epoch the
%     learning rate is multiplied by this value.
%     - batch_size: Size of minibatches used to compute loss and gradient
%     during training.
%     - num_epochs: The number of epochs to run for during training.
%     - print_every: Integer; training losses will be printed every
%     print_every iterations.
%     - verbose: Boolean; if set to false then no output will be printed
%         during training.
%         - num_train_samples: Number of training samples used to check training
%         accuracy; default is 1000; set to None to use entire training set.
%         - num_val_samples: Number of validation samples to use to check val
%         accuracy; default is None, which uses the entire validation set.
%         - checkpoint_name: If not None, then save model checkpoints here every
%         epoch.

self.model = model;
self.X_train = data.X_train;
self.y_train = data.y_train;
self.X_val = data.X_val;
self.y_val = data.y_val;

%  Unpack keyword arguments
% self.update_rule = kwargs.pop('update_rule', 'sgd')
% self.optim_config = kwargs.pop('optim_config', {})
self.lr_decay = opt.lr_decay;
self.model.learning_rate = opt.learning_rate;
self.batch_size = opt.batch_size;
self.num_epochs = opt.num_epochs;
self.num_train_samples = size(self.X_train, 1);
self.num_val_samples = size(self.X_val, 1);

% self.checkpoint_name = kwargs.pop('checkpoint_name', None)
self.print_every = opt.print_every;
self.verbose = opt.verbose;

self = reset(self);
% self = step(self);
self = train(self);
result = self.result;
end

function self = reset(self)
%  Set up some book-keeping variables for optimization. Don't call this manually.
result.epoch = 1;
result.best_loss = 1000000000000;
result.best_params = {};
result.loss_history = [];
result.train_loss_history = [];
result.val_loss_history = [];

self.result = result;

config_W = {};
for i = 1 : length(self.model.params.W)
    x = self.model.params.W{i};
    config.learning_rate = self.model.learning_rate;
    config.beta1 = 0.9;
    config.beta2 = 0.999;
    config.epsilon = 1e-8;
    config.m = zeros(size(x));
    config.v = zeros(size(x));
    config.t = 1;
    config_W = [config_W; config];
end
self.model.config.W = config_W;

config_b = {};
for i = 1 : length(self.model.params.b)
    x = self.model.params.b{i};
    config.learning_rate = self.model.learning_rate;
    config.beta1 = 0.9;
    config.beta2 = 0.999;
    config.epsilon = 1e-8;
    config.m = zeros(size(x));
    config.v = zeros(size(x));
    config.t = 1;
    config_b = [config_b; config];
end
self.model.config.b = config_b;

if self.model.use_batchnorm
    config_gamma = {};
    for i = 1 : length(self.model.params.gamma)
        x = self.model.params.gamma{i};
        config.learning_rate = self.model.learning_rate;
        config.beta1 = 0.9;
        config.beta2 = 0.999;
        config.epsilon = 1e-8;
        config.m = zeros(size(x));
        config.v = zeros(size(x));
        config.t = 1;
        config_gamma = [config_gamma; config];
    end
    self.model.config.gamma = config_gamma;
    
    config_beta = {};
    for i = 1 : length(self.model.params.beta)
        x = self.model.params.beta{i};
        config.learning_rate = self.model.learning_rate;
        config.beta1 = 0.9;
        config.beta2 = 0.999;
        config.epsilon = 1e-8;
        config.m = zeros(size(x));
        config.v = zeros(size(x));
        config.t = 1;
        config_beta = [config_beta; config];
    end
    self.model.config.beta = config_beta;
end

end

function self = step(self, X_batch, y_batch)
% Make a single gradient update. This is called by train() and should not
% be called manually.

% Compute loss and gradient
[loss, grads, self.model] = Compute_loss(self.model, X_batch, y_batch);
self.result.loss_history = [self.result.loss_history; loss];

% Perform a parameter update
for p = 1 : self.model.num_layers
    w = self.model.params.W{p};
    b = self.model.params.b{p};
    dw = grads.W{p};
    db = grads.b{p};
    %     config = self.optim_configs[p]
    %     next_w = sgd(w, dw, self.model.learning_rate);
    %     next_b = sgd(b, db, self.model.learning_rate);
%     self.model.config.W{p}.learning_rate = self.model.learning_rate;
%     self.model.config.b{p}.learning_rate = self.model.learning_rate;
    
    [next_w, self.model.config.W{p}] = optimm(w, dw, self.model.optim, self.model.config.W{p});
    [next_b, self.model.config.b{p}] = optimm(b, db, self.model.optim, self.model.config.b{p});
    self.model.params.W{p} = next_w;
    self.model.params.b{p} = next_b;
    %     self.optim_configs[p] = next_config
    if self.model.use_batchnorm && p < self.model.num_layers
        gamma = self.model.params.gamma{p};
        beta = self.model.params.beta{p};
        dgamma = grads.gamma{p};
        dbeta = grads.beta{p};
        [next_gamma, self.model.config.gamma{p}] = optimm(gamma, dgamma, self.model.optim, self.model.config.gamma{p});
        [next_beta, self.model.config.beta{p}] = optimm(beta, dbeta, self.model.optim, self.model.config.beta{p});
        %         next_gamma = sgd(gamma, dgamma, self.model.learning_rate);
        %         next_beta = sgd(beta, dbeta, self.model.learning_rate);
        self.model.params.gamma{p} = next_gamma;
        self.model.params.beta{p} = next_beta;
    end
end
end

function self = train(self)
%  Run optimization to train the model.
num_train = self.num_train_samples;
iterations_per_epoch = ceil(num_train / self.batch_size);
num_iterations = self.num_epochs * iterations_per_epoch;
X_val = self.X_val;
y_val = self.y_val;

for id = 1 : self.num_epochs
    % training data random arrangement
    id_random = randperm(num_train);
    if num_train < iterations_per_epoch * self.batch_size
        id_random = [id_random,...
            zeros(1, iterations_per_epoch * self.batch_size - num_train)];
    end
    id_part = reshape(id_random, iterations_per_epoch, self.batch_size);
    
    fprintf('****************************************************\n');
    fprintf('(Epoch %d / %d)\n', id, self.num_epochs);
    fprintf('****************************************************\n');
    
    for t = 1 : iterations_per_epoch
        % Make a minibatch of training data
        batch_mask = sort(id_part(t,:));
        batch_mask(batch_mask == 0)=[];
        X_batch = self.X_train(batch_mask,:);
        y_batch = self.y_train(batch_mask,:);
        
        self = step(self, X_batch, y_batch);
        
        %  Save and print the first iteration loss of train and val dataset
        if id * t == 1
            self.result.train_loss_history = [self.result.train_loss_history; self.result.loss_history(end)];
            [loss_val, ~] = Compute_loss(self.model, X_val, y_val);
            self.result.val_loss_history = [self.result.val_loss_history; loss_val];
            if self.verbose
                fprintf('(Iteration %d / %d) train loss: %f; val loss: %f \n',...
                    id * t, num_iterations, self.result.train_loss_history(end), self.result.val_loss_history(end));
            end
        end
        
        %  Maybe print training loss
        if self.verbose && mod(t, self.print_every) == 0
            fprintf('(Iteration %d / %d) loss: %f \n',...
                id * t, num_iterations, self.result.loss_history(end));
        end
    end
    %   At the end of every epoch, increment the epoch counter
    %   and decay the learning rate.
    self.result.epoch = id;
    self.model.learning_rate = self.lr_decay * self.model.learning_rate;
    
    %  Save and print the train and val loss at the end of each epoch.
    self.result.train_loss_history = [self.result.train_loss_history; self.result.loss_history(end)];
    [loss_val, ~] = Compute_loss(self.model, X_val, y_val);
    self.result.val_loss_history = [self.result.val_loss_history; loss_val];
    if self.verbose
        fprintf('(Epoch %d / %d) train loss: %f; val loss: %f \n',...
            id, self.num_epochs, self.result.train_loss_history(end), self.result.val_loss_history(end));
    end
    
    % Keep track of the best model
    if self.result.best_loss > loss_val
        self.result.best_loss = loss_val;
        self.result.best_params = self.model;
    end
end
end

