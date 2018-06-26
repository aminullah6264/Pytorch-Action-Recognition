clc;
addpath(strcat('D:\New Experiments\New Action Recognition\pool3Features'));
for i=1:51
    CFolder=dir(strcat('D:\New Experiments\New Action Recognition\pool3Features\*class_',int2str(i),'.csv'));
    CSize=length(CFolder);
    for j=1:CSize
        if j <= CSize*0.8
            movefile (strcat('D:\New Experiments\New Action Recognition\pool3Features\',CFolder(j).name),'D:\New Experiments\New Action Recognition\pool3Features\Train1\');
        else
            movefile (strcat('D:\New Experiments\New Action Recognition\pool3Features\',CFolder(j).name),'D:\New Experiments\New Action Recognition\pool3Features\Test1\');
        end
    end
end

