%function testImg
%{
skinImg = imageDatastore('skinData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
numObs = length(skinImg.Labels)

numObsPerClass = countEachLabel(skinImg)

load data;
options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(skinImg,layers_1,options);
%}
%testing the network
I = imread('t2.png');
J = imresize(I,[240 240]);
yp = classify(trainedNetwork_2,J);

