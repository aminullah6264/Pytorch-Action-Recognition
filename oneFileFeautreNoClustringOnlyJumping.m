clc
warning off
videoNumber=1;
net = caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel',   'test');
net.blobs('data').reshape([227 227 3 1]);





% DBFolder=dir('D:\Datasets\UCF101\UCF-101');


analysis=zeros(1,3);

Features=zeros(1,1000);
lables=zeros(1,1);

jump=5;
class=0;
MainFolder=dir('D:\Action and Scenes\Code\Arranged Hollywood2\Test Videos');
MSize=length(MainFolder);

for tt=3:MSize
    class=class+1;
%     DBFolder=dir(strcat('D:\Action and Scenes\Code\Arranged Hollywood2\Train Videos\',MainFolder(tt).name));
%     DBSize=length(DBFolder);
%     for z=3:DBSize
    addpath(strcat(strcat('D:\Action and Scenes\Code\Arranged Hollywood2\Test Videos\',MainFolder(tt).name)));
    CFolder=dir(strcat(strcat('D:\Action and Scenes\Code\Arranged Hollywood2\Test Videos\',MainFolder(tt).name),'\*.avi'));
%     addpath(strcat('D:\Datasets\UCF50 Youtube\UCF50','\',DBFolder(z).name));
%     CFolder=dir(strcat('D:\Datasets\UCF50 Youtube\UCF50','\',DBFolder(z).name,'\*.avi'));

%     addpath(strcat('D:\Datasets\UCF101\UCF-101','\',DBFolder(z).name));
%     CFolder=dir(strcat('D:\Datasets\UCF101\UCF-101','\',DBFolder(z).name,'\*.avi'));
    
    CSize=length(CFolder);
        for videoNumber=1:CSize
                path=strcat(strcat('D:\Action and Scenes\Code\Arranged Hollywood2\Test Videos\',MainFolder(tt).name),'\',CFolder(videoNumber).name);
                vidObj = VideoReader(path); 
%                 analysis(k,1)=vidObj.Duration;
%                 analysis(k,2)=vidObj.FrameRate;
                  numFrames=vidObj.NumberOfFrames;
                  k=1;
                for i=1:jump:numFrames-mod(numFrames,30)
                    img=read(vidObj,i);
                    tic
                    im_data = imresize(img, [227 227]);% - mean_data; % resize to 256 x 256 and subtract mean
                    res = net.forward({im_data}); % run forward
%                     tt = res{end}';
%                     figure,
%                     subplot(121),imshow(img);
%                     tt=tt./(max(tt));
%                     subplot(122),imshow(imresize(tt,[10 200])),colormap('HSV');
                    Features(k,:) = res{end}; % get feature
                    toc
                    k=k+1;
                end
                [rr,cc]=size(Features);
                NFeatures=reshape(Features(:,:),[rr/6 6000]);
                filename=strcat('File_',int2str(videoNumber),'class_',int2str(class),'.csv');
                csvwrite(filename,NFeatures);
        end
%     end
end

%%% CLASS NAMES %%%
% 
% MainFolder=dir('D:\Datasets\UCF101\UCF-101');
% MSize=length(MainFolder);
% tt= {};
% i=3
% for i=3:103
%     tt{i}=MainFolder(i).name;
%     fprintf('%s \n',MainFolder(i).name);
% end
% ttt=cell2table(tt');

