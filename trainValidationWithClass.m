warning off;
clc;
addpath(strcat('D:\New Experiments\New Action Recognition\pool3Features\Classes1'));
TrainData=[];
ValData=[];
TrainLables=[];
ValLables=[];

for j=1:6
    File = csvread(strcat('D:\New Experiments\New Action Recognition\pool3Features\Classes1\\Features_class_',int2str(j),'.csv'));
    [rr,cc]=size(File);
    TrainData=[TrainData;File(1:floor(rr*0.8),:)];
    ValData=[ValData;File(rr*0.8:end,:)];
    tt=zeros(floor(rr*0.8),1);
    tt(:)=j;
    TrainLables=[TrainLables;tt];
    vv=zeros(floor(rr-rr*0.8+1),1);
    vv(:)=j;
    ValLables=[ValLables;vv];
    fprintf('class %d\n',j);
end