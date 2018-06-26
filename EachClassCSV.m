clc;
addpath(strcat('D:\New Experiments\New Action Recognition\pool3Features\Train1'));
Features=[];
for i=1:6
    CFolder=dir(strcat('D:\New Experiments\New Action Recognition\pool3Features\Train1\*class_',int2str(i),'.csv'));
    CSize=length(CFolder);
    Features=[];
    for j=1:CSize
        File = csvread(CFolder(j).name);
        Features=[Features;File];
    end
    filename=strcat('D:\New Experiments\New Action Recognition\pool3Features\Classes1\Features_class_',int2str(i),'.csv');
    csvwrite(filename,Features);
end