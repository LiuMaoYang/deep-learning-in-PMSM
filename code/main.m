% coding by Yang mao
% 2018.6.1

clear
close all
clc

[X,y,cache] = dataTrans('TTBL_64');
% plot3(X(:,1), X(:,2), y);

% noise = random('Normal',-1,2, length(X),1);
% y = X.^2 + X + sin(X) - tanh(X) + noise;

% X = randperm(2000);
% X = X';
% y = X;
% y(mod(X,2)==0)=1;
% y(mod(X,2)~=0)=0;

% X = linspace(-200, 200, 1000);
% y = 10 * log(X) .* sign(X) + X;
% plot(X, y)

num_sample = size(X, 1);
id_random = randperm(num_sample);

% Split data to Part.train and Part.val
split_precent = 0.7; % the precent of train data 
id_train = 1:num_sample;
% id_train = sort(id_random(1:round(num_sample * split_precent)));
id_val = sort(id_random(round(num_sample * split_precent) + 1 : end));
    
data.X_train = X(id_train,:);
data.X_val = X(id_val,:);
data.y_train = y(id_train,:);
data.y_val = y(id_val,:);

s.learning_rate = 2e-3;
s.lr_decay = 0.8;
s.batch_size = 360;
s.num_epochs = 500;
s.use_batchnorm = 0;
s.use_dropout = 0;
s.reg = 1e-2;
s.weight_scale=1e-2;
s.eps = 1e-5;
s.momentum = 0.9;
s.print_every = 10;
s.verbose = true;
s.active = 'relu'; % active layer: relu tanh sigmoid
s.optim = 'adam'; % optimization: agd adam

hidden_dims = [30 60 30];
input_dim = size(data.X_train,2); output_dim = size(data.y_train, 2);
model = DLnet(hidden_dims, input_dim, output_dim, s);
result = DLtrain(model, data, s);

p_x = 0:result.epoch;
% figure; plot(p_x, result.train_loss_history);title('train_loss');
% figure; plot(p_x, result.val_loss_history);title('val_loss');
% figure; plot(p_x, result.train_loss_history, p_x, result.val_loss_history); title('loss');
% legend('train_loss', 'val_loss');
% xlabel('epoch'); ylabel('loss');

out = Compute_loss(result.best_params, X);
err = out - y;
figure; surf(cache{1}, cache{2}, reshape(err_,size(cache{1})), 'EdgeColor', 'none'); title('Error'); xlabel('电流'); ylabel('角度'); zlabel('转矩误差');

loss = sum(abs(err))/num_sample;
fprintf('****************************************************\n');
fprintf('loss: %f , max_loss: %f \n',loss, max(abs(err)));
fprintf('****************************************************\n');
% figure; plot(1:length(y), err); title('err');
figure; surf(cache{1}, cache{2}, reshape(out, size(cache{1})), 'EdgeColor', 'none'); xlabel('电流'); ylabel('角度'); zlabel('转矩');
figure; plot3(X(:,1), X(:,2), y, X(:,1), X(:,2), out); xlabel('电流'); ylabel('角度'); zlabel('转矩');
% figure; plot3(X(:,1), X(:,2), err); title('Error'); xlabel('电流'); ylabel('角度'); zlabel('转矩误差');
% % 
% t = 3;
% err_ = err;
% err_(abs(err)>t) = rand(size(err_(abs(err)>t))) .* sign(err_(abs(err)>t));
% figure; surf(cache{1}, cache{2}, reshape(err_,size(cache{1})), 'EdgeColor', 'none'); title('Error'); xlabel('电流'); ylabel('角度'); zlabel('转矩误差');
% % 
err = reshape(err,size(cache{1}));
figure; surf(cache{1}, cache{2}, err);


% plot(X, y, X, out)
% error = out - y;
% figure; plot(1:num_sample, error);
% test = randi(10,1,5);
% out = Compute_loss(result.best_params, test')