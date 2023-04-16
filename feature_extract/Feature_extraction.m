clear all
clc

path = 'E:\Lab\EOG_Code\infant_sleep - tf20-2\input prepare\true_ouput\multi-crowd\';
load('data.mat');

% PSD extraction
N = size(x,2);
nfft = 4096;
fs = 128;
window = boxcar(1024);
noverlap = 128;
[pxx,w] = pwelch(x',window,noverlap,nfft,fs);

all_psd = pxx';
gap = fs/nfft;
f_psd = zeros(size(x,1),5);
for i = 1:size(all_psd,1)
    %delta
    f_psd(i,1) = sum(all_psd(i,0.5/gap+1:4/gap+1));
    %sita
    f_psd(i,2) = sum(all_psd(i,4/gap+1:8/gap+1));
    %alpha
    f_psd(i,3) = sum(all_psd(i,8/gap+1:13/gap+1));
    %beta
    f_psd(i,4) = sum(all_psd(i,13/gap+1:20/gap+1));
    %gama
    f_psd(i,5) = sum(all_psd(i,20/gap:-1));
end
%功率
p = mean(abs(x').^2);

%能量
e = sum(abs(x').^2);

%kurtosis
kurt = kurtosis(x');

%skewness
ske = skewness(x');

x0 = [f_psd, p', e']; 

save(['E:\Lab\EOG_Code\infant_sleep - tf20-2\input prepare\true_ouput\multi-crowd/data_feature.mat'], 'x0', 'y');

