inputSize = [240 240 3];
imds = imageDatastore('skinData',IncludeSubfolders=true,LabelSource="foldernames");
[imdsTest,imdsValidation] = splitEachLabel(imds,0.5,"randomize");
augmenter = imageDataAugmenter(RandXReflection=true,RandScale=[0.5 1.5]);
%augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,DataAugmentation=augmenter);
augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation);
%optsSmallnet = trainingOptions("sgdm", MaxEpochs=40, InitialLearnRate=0.01, ValidationData=augimdsValidation, ValidationFrequency=50, Verbose=false, Plots="training-progress");
augimdsTest = augmentedImageDatastore(inputSize,imdsTest);
[YTestSmallNet,scoresSmallNet] = classify(trainedNetwork_3,augimdsTest);
%etSmallNet = trainNetwork(augimdsTrain,layerData,optsSmallNet);
analyzeNetwork(trainedNetwork_3)
TTest = imdsTest.Labels;
accSmallNet = sum(TTest == YTestSmallNet)/numel(TTest)
classNames = trainedNetwork_3.Layers(end).Classes;
rocSmallNet = rocmetrics(TTest,scoresSmallNet,classNames);
figure
plot(rocSmallNet,ShowModelOperatingPoint=false)
legend(classNames)
title("ROC Curve: Malignant")
aucSmallNet = rocSmallNet.AUC;
print("AUC=", aucSmallNet)
