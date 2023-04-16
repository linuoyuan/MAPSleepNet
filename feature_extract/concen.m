clear all
clc;

path1 = 'E:\Lab\EOG_Code\infant_sleep - tf20-2\input prepare\true_ouput\multi-crowd\infant\';
path2 = 'E:\Lab\EOG_Code\infant_sleep - tf20-2\input prepare\true_ouput\multi-crowd\teenager\';
path3 = 'E:\Lab\EOG_Code\infant_sleep - tf20-2\input prepare\true_ouput\multi-crowd\adult\';

list1 = dir([path1,'*mat']);
list2 = dir([path2,'*mat']);
list3 = dir([path3,'*mat']);

x1 = []; %infant features
x2 = []; %tennager
x3 = []; %adult
label = 0;
stopnum = 20;

for i1 = 1:numel(list1)
    disp(list1(i1).name);
    load([path1,list1(i1).name]);
    sign = 0;
    for l = 1:numel(y_5)
%         if y_5(l) == label
            sign = sign+1;
            x1(sign + i1*stopnum - stopnum,:) = FE(xx(l,:));
%         end
        if sign == stopnum
            break;
        end
    end
%     x1(i1,:) = reshape(xx(1:10,:),1,[]);
end
y1 = ones(size(x1,1),1);

for i2 = 1:numel(list2)
    disp(list2(i2).name);
    load([path2,list2(i2).name]);
    sign = 0;
    for l = 1:numel(y_5)
%         if y_5(l) == label
            sign = sign+1;
            x2(sign + i2*stopnum - stopnum,:) = xx(l,:);
%         end
        if sign == stopnum
            break;
        end
    end
% 	x2(i2,:) = reshape(xx(1:10,:),1,[]);
end
y2 = 2*ones(size(x2,1),1);

for i3 = 1:numel(list3)
    disp(list3(i3).name);
    load([path3,list3(i3).name]);
    sign = 0;
    for l = 1:numel(y_5)
%         if y_5(l) == label
            sign = sign+1;
            x3(sign + i3*stopnum - stopnum,:) = xx(l,:);
%         end
        if sign == stopnum
            break;
        end
    end
%     x3(i3,:) = reshape(xx(1:10,:),1,[]);
end
y3 = 3*ones(size(x3,1),1);

x = [x1;x2;x3];
y = [y1;y2;y3];
save(['E:\Lab\EOG_Code\infant_sleep - tf20-2\input prepare\true_ouput\multi-crowd/data.mat'], 'x', 'y');

% N = size(x,2);
% nfft = 4096;
% fs = 128;
% window = boxcar(1024);
% noverlap = 128;
% [pxx,w] = pwelch(x',window,noverlap,nfft,fs);
% 
% 
% all_psd = pxx';
% gap = fs/nfft;
% f_psd = zeros(size(x,1),5);
% for i = 1:size(all_psd,1)
%     %delta
%     f_psd(i,1) = sum(all_psd(i,0.5/gap+1:4/gap+1));
%     %sita
%     f_psd(i,2) = sum(all_psd(i,4/gap+1:8/gap+1));
%     %alpha
%     f_psd(i,3) = sum(all_psd(i,8/gap+1:13/gap+1));
%     %beta
%     f_psd(i,4) = sum(all_psd(i,13/gap+1:20/gap+1));
%     %gama
%     f_psd(i,5) = sum(all_psd(i,20/gap:numel(w)));
% end
% 
% %功率
% p = mean(abs(x').^2);
% 
% %能量
% e = sum(abs(x').^2);
% 
% %kurtosis
% kurt = kurtosis(x');
% 
% %skewness
% ske = skewness(x');
% 
% x0 = [f_psd, p', e']; 
% 
% save(['E:\Lab\EOG_Code\infant_sleep - tf20-2\input prepare\true_ouput\multi-crowd/data_feature.mat'], 'x0', 'y');

function x0 = FE(xx)
    N = size(xx,2);
    nfft = 4096;
    fs = 128;
    window = boxcar(1024);
    noverlap = 128;
    [pxx,w] = pwelch(xx',window,noverlap,nfft,fs);

    all_psd = pxx';
    gap = fs/nfft;
    f_psd = zeros(size(xx,1),5);
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
    p = mean(abs(xx').^2);

    %能量
    e = sum(abs(xx').^2);

    %kurtosis
    kurt = kurtosis(xx');

    %skewness
    ske = skewness(xx');

    x0 = [f_psd, p', e']; 
end




