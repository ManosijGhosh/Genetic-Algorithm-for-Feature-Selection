clear all
x=importdata('Train.csv');   % Training File 
t=importdata('TrainTargets.csv');  % Training label
Test=importdata('Test.csv');   % Test File
TestTargets=importdata('TestTargets.csv');   % Test Label
% load ('allFeature');
% load ('totalTrainLabel');
allFeature=x(:,1:288);
totalTrainLabel=t;
% save Test Test;
% save TestTargets TestTargets;
numberOfFold=4;
numberOfClass=80;
numberOfFeature=size(allFeature,2);
numberOfData=size(allFeature,1);
numberOfSamplePerClass=numberOfData/numberOfClass;
disp([numberOfData numberOfSamplePerClass numberOfFeature]);
trainRow=floor((numberOfFold-1)*(numberOfData/numberOfFold));
testRow=numberOfData-trainRow;
per=0;
for hddnlayer=70:70
    for fold=1:4
        trainData=zeros(trainRow,numberOfFeature);
        testData=zeros(testRow,numberOfFeature);
        trainLabel=zeros(trainRow,numberOfClass);
        testLabel=zeros(testRow,numberOfClass);
        tec=1;
        trc=1;
        for i=1:numberOfClass
            for j=1:numberOfSamplePerClass
                t1=(fold-1)*(numberOfSamplePerClass/numberOfFold);
                t2=fold*(numberOfSamplePerClass/numberOfFold);
                if(j>t1 && j<=t2)                              %  k>=fold*(NOSPC/FN) && k<(NOSPC/FN)*(fold+1)
                    testData(tec,:)=allFeature((i-1)*numberOfSamplePerClass+j,:);
                    testLabel(tec,:)=totalTrainLabel((i-1)*numberOfSamplePerClass+j,:);
                    tec=tec+1;
                else
                    trainData(trc,:)=allFeature((i-1)*numberOfSamplePerClass+j,:);
                    trainLabel(trc,:)=totalTrainLabel((i-1)*numberOfSamplePerClass+j,:);
                    trc=trc+1;
                end
            end
        end
        disp('Data Created');
%         TrainTest=testData;
%         save TrainTest TrainTest;
%         TrainTrain=trainData;
%         save TrainTrain TrainTrain;
%         TrainTestTargets=testLabel;
%         save TrainTestTargets TrainTestTargets;
%         TrainTrainTargets=trainLabel;
%         save TrainTrainTargets TrainTrainTargets;
%         
        % trainData=allFeature();
        hiddenLayerSize = hddnlayer;  %determins the umber of layers and neurons in hidden layers
        net = patternnet(hiddenLayerSize);
        inputs = trainData';
        targets = trainLabel';
        countt=0;
        while (countt<=0)
        % Setup Division of Data for Training, Validation, Testing
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 30/100;
        net.divideParam.testRatio = 0/100;
        % Train the Network
        [net, ] = train(net,inputs,targets);
        % Test the Network
        % input = inputs(:,1:100);
        input = testData';
        % target =targets(:,1:100);
        target =testLabel';
        % targets = t2';
        outputs = net(input);
%         save output outputs;
        %outputs
        [c, ] = confusion(target,outputs);
        performance=100*(1-c);
        % fprintf('The number of features  : %d\n', sum(chromosome(:)==1));
%         fprintf('Percentage Correct Classification at %d th : %f\n',countt, performance);
        if(performance>per)
            per=performance;
            save net net;
            fprintf('Net saved for fold %d at hidden layer %d, count=%d with performance %f\n',fold, hddnlayer,countt, per);
        end
        countt=countt+1;
        end
    end
end
load net;
Test1=Test(:,1:288);
Test=Test1';
outputs = net(Test);
[c, ] = confusion(TestTargets',outputs);
performance=100*(1-c);
fprintf('Performance on Test=%f\n',performance);