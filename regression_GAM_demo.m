clear all;
clc
% Load data
dat = csvread('concrete_train.csv',1);
y = dat(:,9);
X = dat(:,1:8);

% Split the Data into Test and Training 
rand('seed',5);
fold = 10;
ii = repmat(1:fold, 1, size(dat,1)/fold);
ind = randperm(length(ii));
ii = ii(ind);
Xtrain = X(ii ~= 1,:);
Xtest = X(ii == 1,:);
ytrain = y(ii~=1); 
ytest = y(ii==1);

[nTrain, nFeatures] = size(Xtrain);
options.degree = 3;
options.knots = 3000;
options.lambda = 200;
options.nboost = 3000;

% Basic Demo, Evolution of the Error
model = matLearn_regression_GAM(Xtrain,ytrain,options,@regression_spline );
error = model.error;
figure(1);
plot(1:100:model.nboost, error, '-ro');
title('Evolution of Error, Training Set')
xlabel('iterations')
ylabel('residual error')
minErr.train = min(error);

figure(2);
[f.test, test_error] = model.predict(model,Xtest,ytest,options);
plot(1:100:model.nboost, test_error, '-ro');
title('Evolution of Error, Testing Set')
xlabel('iterations')
ylabel('residual error')
minErr.test = min(test_error)


