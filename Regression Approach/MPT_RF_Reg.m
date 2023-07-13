clc
close all
warning off

numStocks = 10;
numVariable = 7;
numDays = 2516;
testPortion = 0.2;

% LOAD DATA

stocks = ["AAPL";"AMD";"AMZN";"CSCO";"META";"MSFT";"NFLX";"QCOM";"SBUX";"TSLA"];
dataAAPL = readmatrix("Data/AAPL.csv");
dataAMD = readmatrix("Data/AMD.csv");
dataAMZN = readmatrix("Data/AMZN.csv");
dataCSCO = readmatrix("Data/CSCO.csv");
dataMETA = readmatrix("Data/META.csv");
dataMSFT = readmatrix("Data/MSFT.csv");
dataNFLX = readmatrix("Data/NFLX.csv");
dataQCOM = readmatrix("Data/QCOM.csv");
dataSBUX = readmatrix("Data/SBUX.csv");
dataTSLA = readmatrix("Data/TSLA.csv");

% COMPILE IN ONE MATRIX

data = zeros(numStocks, numDays, numVariable);
data(1,:,:) = dataAAPL; data(2,:,:) = dataAMD; data(3,:,:) = dataAMZN;
data(4,:,:) = dataCSCO; data(5,:,:) = dataMETA; data(6,:,:) = dataMSFT;
data(7,:,:) = dataNFLX; data(8,:,:) = dataQCOM; data(9,:,:) = dataSBUX;
data(10,:,:)= dataTSLA;
%plot(data(1, :, 1))
%hold on

% EXPONENTIAL SMOOTHING OF DATA

alpha = 0.2;
for i = 1:numStocks
    for j = 1:numVariable
        for k = 2:numDays-1
            data(i,k,j) = (alpha * data(i, k, j)) + ((1-alpha) * data(i, k-1, j));
        end
    end
end
%plot(data(1, :, 1))
%hold on

% BUILD RANDOM FOREST

% Accuracy and error arrays of stocks after predicting stock direction
stockAccur = zeros(numStocks, 1);
stockError = zeros(numStocks, 1);
%Matrices to store predicted and actual returns on test days
returnPredicted = zeros(floor(testPortion*numDays), numStocks);
returnActual = zeros(floor(testPortion*numDays), numStocks);
for i = 1:numStocks
    dataTemp = zeros(numDays, numVariable);
    for j = 1:numDays
        for k = 1:numVariable
            dataTemp(j,k) = data(i,j,k);
        end
    end
    cv = cvpartition(size(dataTemp,1),'HoldOut',testPortion);   %Partion of days
    idx = cv.test;
    dataTrain = dataTemp(~idx, :);
    dataTest = dataTemp(idx,:);
    testing = dataTest(1:end,1:end-1);
    %Train and test model
    model = fitensemble(dataTrain(:, 1:end-1), dataTrain(:, end),'Bag',100,'Tree','Type','regression');
    prediction = predict(model,testing);
    returnPredicted(:, i) = prediction;
    returnActual(:, i) = dataTest(:, end);
  
    error = norm(prediction - dataTest(:,end));
    stockError(i, 1) = error;
end
disp(stockError);
maxError = max(stockError(1:end, 1));
for i = 1:numStocks
    stockAccur(i,1) = 100 * ((abs(stockError(i,1) - maxError)) / maxError);
end
disp(stockAccur);

plot(returnPredicted(:, 4)', 'r');
%xlim([400, 510]);
hold on
plot(returnActual(:, 4)', 'b');
xlabel 'Days'; ylabel 'Returns(%)';
title('Predicted and Actual Returns');
legend({'Predicted Return(%)','Actual Return(%)'}, 'Location','Best');

% RANK STOCKS ON THE BASIS OF ACCURACY

[rank, idx] = sort(stockAccur, "descend");
rankedStocks = stocks(idx,:)




