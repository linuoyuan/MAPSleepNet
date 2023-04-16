clear all
clc

path = 'E:\Lab\EOG_Code\infant_sleep - tf20-2\input prepare\true_ouput\multi-crowd\';
load('data_feature.mat');

rowrank = randperm(size(x0, 1)); 
x0 = x0(rowrank,:); 
y = y(rowrank);

len = size(x0,1);
traindata = x0(1:0.8*len,:);
testdata = x0(0.8*len:len,:);
trainlabel = y(1:0.8*len);
testlabel = y(0.8*len:len);
k = 3;
x = traindata;
Mdl = KDTreeSearcher(x);
[n,~] = knnsearch(Mdl,testdata,'k',k);
for i = 1:size(n,1)
 
    tempClass = trainlabel;
    result = mode(tempClass);
    resultClass(i,1) = result;
 
end
 
validate = sum( testlabel == resultClass )./ size(testlabel,1) * 100;