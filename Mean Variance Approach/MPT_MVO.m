clc
close all
warning off

numStocks = 10;
numVariable = 7;
numDays = 2516;

% LOAD DATA
returnData = readmatrix("Data/Returns.xlsx");
stocks = ["AAPL","AMD","AMZN","CSCO","META","MSFT","NFLX","QCOM","SBUX","TSLA"];

% MEAN OF HISTORICAL DATA AS RETURN AND COVARIANCE AS RISK
meanReturn = mean(returnData);
%Choose best assets with most return
numStocks = 5;  %we want to invest in only 5 stocks
[meanReturnNew, idx] = sort(meanReturn, "descend");   %Sort returnData
meanReturnData = meanReturnNew(:,1:numStocks);  %Top 5 returns
stocksSorted = stocks(:,idx);
stocksToInvest = stocksSorted(:,1:numStocks);
cov_mat = cov(meanReturnData);

%Initialize the storage variables
numIter = 50;
requiredReturn_mat = zeros(numIter,1);
weights_mat = zeros(numIter, numStocks);
var_mat = zeros(numIter, 1);
risk_mat = zeros(numIter, 1);
sharpeRatio_mat = zeros(numIter,1);

%Call function for different values of requiredReturn
for i = 1:numIter
    requiredReturn = 0.055 + (i-1)*0.0004;
    [w, v, sr] = solve(requiredReturn, numStocks, meanReturnData, cov_mat);
    weights_mat(i,:) = w;
    var_mat(i,:) = v;
    risk_mat(i, :) = sqrt(v);
    sharpeRatio_mat(i,:) = sr;
    requiredReturn_mat(i,:) = requiredReturn;
end

%Merge requiredReturn, variance and sharpeRatio od eacg iteration
final = [requiredReturn_mat, risk_mat, sharpeRatio_mat];
%disp(final);

%Finding Optimal Sharpe Ratio and corresponding weight
[sharpeMax, sharpeMaxIdx] = max(sharpeRatio_mat);
[weightOptimized] = weights_mat(sharpeMaxIdx,:);
disp(weightOptimized);

%Plot risk-return graph
x = final(:,2); xPlot = x'; xPloti = xPlot;
y = final(:,1); yPlot = y';
subplot(1,2,1);
plot(xPlot,yPlot); xlabel 'Standard Deviation(Risk)'; ylabel 'Return';
xlim([0.002, 0.009]); ylim([0.045, 0.085]);
title 'Risk-Return';
%Highlight point related to maximum sharpeRatio
hold on;
plot(final(sharpeMaxIdx, 2), final(sharpeMaxIdx, 1), 'o','MarkerSize',6);
%Plot sharpeRatios
subplot(1,2,2);
plot(sharpeRatio_mat); xlabel 'Iterations'; ylabel 'Sharpe Ratio';
xlim([-10, 60]); ylim([6, 18]);
title 'Sharpe Ratio';
%Highlight maximum point
hold on
plot(sharpeMaxIdx, sharpeMax,'o','MarkerSize',5);

%Export matrices into excel sheets
writematrix(meanReturn, 'mean of returns.xlsx');
writematrix(cov_mat, 'covarience of returns.xlsx');
writematrix(final, 'required_return with related risk and sharpeRatio.xlsx');

function [w, v, sr] = solve(requiredReturn, numStocks, meanReturnData, cov_mat)
% SET UP THE CONSTRAINTS
fun = @(w) w * cov_mat * w';    %Function to minimize
%w is weight matrix, w'*cov_mat*w is variance
wo = zeros(1,numStocks);   %Initial guess for weights
%Equality constraints, Aeq * w = Beq
Aeq = [1,1,1,1,1; meanReturnData];
Beq = [1, requiredReturn];
%Inequality constraints, -> no inequality constraints
A = [];
B = [];
%Bounds of weights
lb = zeros(1,numStocks);    %No less than 0 weights to avoid short selling
ub = 0.5 * ones(1,numStocks);   %I want no more than 40% allocation to any asset

%Optimizing function
[weights, varStocks] = fmincon(fun, wo, A, B, Aeq, Beq, lb, ub);
sd = sqrt(varStocks);
sharpeRatio = requiredReturn / sd;

w = weights; v = varStocks; sr = sharpeRatio;
end



