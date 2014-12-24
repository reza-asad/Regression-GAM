function [model] = matLearn_regression_GAM(X,y,options,boostedModel)
% matLearn_regression_GAM(X,y,options)
%
% Description:
%       - Fits a general additive model using gradient boosting
%
% Options:
%       - nboost: The number of boosting iterations (default: 1000)
%
% Author: Reza Asad (2014)

[nTrain,nFeatures] = size(X);

% Default Value for nboost
[nboost] = myProcessOptions(options,'nboost',5000);

f = zeros(nTrain,1);
best = zeros(1,nboost);
Y = repmat(y,1,nFeatures);
v = 0.9;
n = 0;
for i=1:nboost
    % Compute the Negative Gradient and Evaluate ath the Fit
    gradient = -2/nTrain * (f-y);
    
    % Estimate the Negative Gradient Using the Base Learners
    submodel = boostedModel(X,gradient,options);

    % Select the Best Base Learner
    diff = (submodel - Y).^2;
    err = sum(diff,1)/nTrain;
    best(i) = find( err == min(err));
    f_i = submodel(:,best(i));
  
    % Move in the Direction of Best Learner
    if i>500
        v = 0.5;
    end
    f = f + v*f_i;
    
    % Returns the Squared Error at Every 100 Iterations.
    if mod(i,100) == 0
        n = n+1;
        error(n) = 1/nTrain * sum((y - f).^2);
    end

end

model.error = error;
model.best = best;
model.boostedModel = boostedModel;
model.nboost = nboost;
model.predict = @predict;

end

function [f,test_error] = predict(model,Xtest,ytest,options)
[nTest,nFeatures] = size(Xtest);

f = zeros(nTest,1);
v = 0.9;
n = 0;
for i=1:model.nboost
    gradient = -2/nTest * (f-ytest);
    submodel = model.boostedModel(Xtest,gradient,options);
    f_i = submodel(:,model.best(i));
    if i>500
        v = 0.5;
    end
    f = f+v*f_i;
    if mod(i,100) == 0
        n = n+1;
        test_error(n) = 1/nTest * sum((ytest - f).^2);
    end
end

end



