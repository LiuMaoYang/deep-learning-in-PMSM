function [input,target,cache] = dataTrans(fileName)
data = xlsread(fileName);

gridX = data(1,2:end);
gridY = data(2:end,1);
[x,y] = meshgrid(gridX, gridY);
input = [x(:) y(:)];

[~,c]=size(data);
target=zeros(0,1);
for i=2:c
    target = [target;data(2:end,i)];
end
cache = {x, y, data(2:end,2:end)};

% xlswrite('input.xlsx',input);
% xlswrite('target.xlsx',target);
end
