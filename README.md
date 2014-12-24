Regression-GAM
==============
Given N variables (features), I preform spline regression on each variable (see 'regression_spline.m'). These are called the weak learners. In order to make a powerful regrression model, I use gradient boosting to combine the weak learners togeather. Moreover, I have provided a script ('regression_GAM_demo.m') where you can see the evolution of the test and training error with respect to the boosting iterations.   
