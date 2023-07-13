clc
close all
warning off

numStocks = 10;
numVariable = 7;
numDays = 2516;
testPortion = 0.25;

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

%{
% PLOT VARIABLES OF EACH STOCK
for i = 1:numStocks
    figure('Name','VARIABLES of stock');
    subplot(2,2,1); plot(data(i, :, 1)); title('Historical On Balance Volume');
    subplot(2,2,2); plot(data(i, :, 4)); title('Historical Average price');
    subplot(2,2,3); plot(data(i, :, 5)); title('Historical Closing Price');
    subplot(2,2,4); plot(data(i, :, 7)); title('Historical Returns(%)');
end
%}

%Demonstration of Exponential Smoothing
figure
subplot(1,2,1); plot(data(1,:,5)); title('Un-Smoothed closing prices of AAPL');

% EXPONENTIAL SMOOTHING OF DATA

alpha = 0.2;
for i = 1:numStocks
    for j = 1:numVariable
        for k = 2:numDays-1
            data(i,k,j) = (alpha * data(i, k, j)) + ((1-alpha) * data(i, k-1, j));
        end
    end
end
subplot(1,2,2); plot(data(1,:,5)); title('Smoothed closing prices of AAPLE');

% FEATURE EXTRACTION

%1. ON BALANCE VOLUME(OBV); Volume is in column 1;
OBV = zeros(numStocks, numDays);
for i = 1:numStocks
    for k = 2:numDays
        if k == 1
            OBV(i, k) = data(i, k, 1);
        end
        if k > 1 && data(i, k, 4) > data(i, k-1, 4) %Price is in column 4
            OBV(i, k) = OBV(i, k-1) + data(i, k, 1);
        end
        if k > 1 && data(i, k, 4) < data(i, k-1, 4)
            OBV(i, k) = OBV(i, k-1) - data(i, k, 1);
        end
        if k > 1 && data(i, k, 4) == data(i, k-1, 4)
            OBV(i, k, 1) = OBV(i, k-1);
        end
    end
end
%plot(OBV(1,:))

% 2. STOCHASTIC OSCILLATOR
StchOsc = zeros(numStocks, numDays);
for i = 1:numStocks
    for k = 1:numDays
        if k-13 > 1
            StchOsc(i, k) = 100 * ((data(i,k,4) - min(data(i,k-13:k,4))) / (max(data(i,k-13:k,4)) - min(data(i,k-13:k,4))));
        end
        if k-13 <= 1
            StchOsc(i, k) = 100 * ((data(i,k,4) - min(data(i,1:k,4))) / (max(data(i,1:k,4)) - min(data(i,1:k,4))));
        end
    end
end
%plot(StchOsc(6,:)); ylim([-20, 120]);

% 3. MOVING AVERAGE CONVERGANCE DIVERGANCE (MACD)
MACD = zeros(numStocks, numDays);
for i = 1:numStocks
    for k = 1:numDays
        if k-25 > 1
            MACD(i, k) = mean(data(i,k-11:k,5)) - mean(data(i,k-25:k,5));   %Closing price is in 5th column
        end
        if k-25 < 1 && k-11 >= 1
            MACD(i, k) = mean(data(i,k-11:k,5)) - mean(data(i,1:12,5));
        end
        if k-11 < 1
            MACD(i, k) = mean(data(i,1:12,5)) - mean(data(i,1:26,5));
        end
    end
end
%plot(MACD(5,:), 'r')
%hold on

% 4. SIGNAL MACD
sMACD = zeros(numStocks, numDays);
for i = 1:numStocks
    for k = 1:numDays
        if k-8 >= 1
            sMACD(i, k) = mean(MACD(i, k-8:k));   % sMACD is 9 period EMA of MACD
        end
        if k-8 < 1
            sMACD(i, k) = mean(MACD(i, 1:9));
        end
    end
end
%plot(sMACD(5,:), 'g');

%{
% PLOT FEATURES OF EACH STOCK
for i = 1:numStocks
    figure('Name','Features of stock', 'NumberTitle','off');
    subplot(2,2,1); plot(data(i, :, 7)); title('Historical Returns(Smoothed)');
    subplot(2,2,2); plot(OBV(i, :)); title('On Balance Volume');
    subplot(2,2,3); plot(StchOsc(i, :)); title('Stochastic Oscillator');
    subplot(2,2,4); plot(MACD(i, :)); title('Moving Average Convergence Divergence');
end
%}

% BUILD RANDOM FOREST

% Accuracy, Precision, Recall and F1 Score arrays of stocks after predicting stock direction
stockAccur = zeros(numStocks, 1);
stockPrecision = zeros(numStocks, 1);
stockRecall = zeros(numStocks, 1);
stockF1Score = zeros(numStocks, 1);
%Matrices to store actual and predicted class(0 or 1)
actualClass = zeros(floor(testPortion*numDays), numStocks);
predictedClass = zeros(floor(testPortion*numDays), numStocks);

figure
for i = 1:numStocks     %Main for loop to compute for each stock individually and store in matrix
    dataTemp = zeros(numDays, numVariable);
    for j = 1:numDays
        for k = 1:numVariable
            dataTemp(j,k) = data(i,j,k);
        end
    end
    %Assign 0 for down and 1 for up prediction
    for days = 1:numDays
        dataTemp(days, end) = dataTemp(days,end) == abs(dataTemp(days,end));
    end
    %dataTable = array2table(dataTemp);    %Convert matrix to table
    cv = cvpartition(size(dataTemp,1),'HoldOut',testPortion);   %Partion of days
    idx = cv.test;
    dataTrain = dataTemp(~idx, :);
    dataTest = dataTemp(idx,:);
    testing = dataTest(1:end,1:end-1);    %Exclude column of Return because that is to be predicted
    %Train and test model
    model = fitensemble(dataTrain(:, 1:end-1), dataTrain(:,end),'Bag',100,'Tree','Type','classification');
    prediction = predict(model,testing);

    %Store Result
    actualClass(:, i) = dataTest(:,end);
    predictedClass(:, i) = prediction;
    %Plot results
    subplot(3, 4, i);
    plot((abs(predictedClass(:, i) - actualClass(:, i)))');
    ylim([-2, 2]); xlim([450, 475]) %using small portion for better visualisation

    %Accuracy
    accuracy = 100 * (sum(prediction == (dataTest(:,end))) / size(dataTest,1));
    stockAccur(i, 1) = accuracy;

    %dataTest = table2array(dataTest);
    
    %Precision of 0 class
    tp = 0; fp = 0;
    for t = 1:size(prediction,1)
        if prediction(t,1) == 0 && dataTest(t,end) == 0
            tp = tp + 1;
        end
        if prediction(t,1) == 0 && dataTest(t,end) == 1
            fp = fp + 1;
        end
    end
    precision0 = tp/(tp+fp);
    %Precision of 1 class
    tp = 0; fp = 0;
    for t = 1:size(prediction,1)
        if prediction(t,1) == 1 && dataTest(t,end) == 1
            tp = tp + 1;
        end
        if prediction(t,1) == 1 && dataTest(t,end) == 0
            fp = fp + 1;
        end
    end
    precision1 = tp/(tp+fp);
    precision = (precision1 + precision0) /2;
    stockPrecision(i,1) = precision*100;
    %Recall of 0 class
    tp = 0; fn = 0;
    for t = 1:size(prediction,1)
        if prediction(t,1) == 0 && dataTest(t,end) == 0
            tp = tp + 1;
        end
        if prediction(t,1) == 1 && dataTest(t,end) == 0
            fn = fn + 1;
        end
    end
    recall0 = tp/(tp+fn);
    %Recall of 1 class
    tp = 0; fn = 0;
    for t = 1:size(prediction,1)
        if prediction(t,1) == 1 && dataTest(t,end) == 1
            tp = tp + 1;
        end
        if prediction(t,1) == 0 && dataTest(t,end) == 1
            fn = fn + 1;
        end
    end
    recall1 = tp/(tp+fn);
    recall = (recall0 + recall1) / 2;
    stockRecall(i,1) = recall*100;
    %F1 score
    stockF1Score(i,1) = (2*stockPrecision(i,1)*stockRecall(i,1)) / (stockPrecision(i,1)+stockRecall(i,1));
    %disp([prediction, dataTest(:,end)]);
end
disp(stockAccur);
disp(stockPrecision);
disp(stockRecall);
disp(stockF1Score);

% CHOOSE BEST PERFORMING STOCKS

% ESTIMATION OF EXPECTED RETURNS

%Volatility using GARCH(1,1) model

% we predict movement of each stock of last 20 days
% then multiply be magnitude in change to get absolute return
figure
for i = 1:numStocks
    numTrainDays = 2500; numPredictDays = 16;
    
    price = data(i,1:numTrainDays,7)';
    %plot(price'); hold on;
    Md1 = garch('GARCHLags', 1, 'ARCHLags', 1, 'Offset', NaN);
    EstMdl = estimate(Md1, price);
    priceF = forecast(EstMdl, numPredictDays, price);
    v = infer(EstMdl, price);
    subplot(3,4,i);
    %Plot actual volatality from day 1 to 80
    plot(1:numTrainDays,v,'k:','LineWidth',2); xlim([2400, numTrainDays+25]);
    hold on;
    % Forecasted volatality from day 81 to 100
    plot(numTrainDays:numTrainDays+numPredictDays,[v(end);priceF],'r:','LineWidth',2);
    xlabel 'Days'; ylabel 'conditional variances';
    title('For. Cond. Var. of Nominal Ret.');
    legend({'Estimation sample cond. var.','Forecasted cond. var.'}, 'Location','Best');
end




