function [model] = regression_spline(X,y,options)
% regression_spline(X,y,options)
%
% Description
%       - Fits a regression spline of degree m
%
% Options: 
%       - degree: This is degree of the spline (default: 2)
%       - nknots: Number of knots (default: 20)
%       - lambda: Penalizes complexity of fit
%
% Author:
%       - Reza Asad (2014)

[nTrain,nFeatures] = size(X);

% Default Values for Knots and Degrees
[degree,nknots,lambda] = myProcessOptions(options,'degree',2, 'nknots',20, 'lambda', 5);

%degree
model = zeros(nTrain,nFeatures);
for j=1:nFeatures
    % Place the Knots
    ran = range(X(:,j));
    knots = zeros(1,nknots);
    for i=1:nknots    
        knots(i) = min(X(:,j)) + i/(nknots+1) * ran;
    end
    % Compute the Design Matrix
    Z = zeros(nTrain,degree+nknots);
    for i=1:degree
        Z(:,i) = X(:,j).^i;
    end
    for i=1:nknots
        Z(:,i+degree) = max((X(:,j)-knots(i)).^degree,0);
    end
    
    % Finds the Regression Fit
    I = eye(degree+nknots);
%     v = ones(degree+nknots,1);
%     v(1:degree) = 0;
%     I = diag(v);
    model(:,j) = Z*((Z'*Z + lambda*I)\Z'*y);
 
end

end

