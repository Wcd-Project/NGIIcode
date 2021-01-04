clear;
clear all;
close all;
addpath('I:\ʵ���Ҳɼ�\512.512RGB-����/');
% queryFile = './queryImgs.txt';
% classesFile = './databaseClasses.txt';
queryFile = './queryImgs1.txt';
classesFile = './databaseClasses1.txt';
% load feat4096Norml-f.mat
% load feat4096Norml-mRGB.mat
% load feat4096Norml-16.mat
% load feat4096Norml-16RGB.mat
% load feat4096Norml-19.mat
% % load feat4096Norml-sRGB.mat
% load feat4096Norml-16RGB-m1024.mat   
% load feat4096Norml-16RGB-m256.mat  
% load feat4096Norml-16RGB-f256.mat 
% load feat4096Norml-16RGB-161024.mat   %256
% load feat4096Norml-16RGB-1610244.mat   %19 1024
% load feat4096Norml-16RGB-19256.mat   %19 1024
% load feat4096Norml-f512.mat

% load feat4096Norml-16RGB-5120001.mat
% load feat4096Norml-ResNet50-512.mat
% load feat4096Norml-19RGB-512000.mat
% load A_FDXJ.mat
load AB_FDXJ.mat
feat_norm=AB;

A=load('A_FDXJ.mat');
B=load('B_FDXJ.mat');
% feat_norm=0.01*A.A+20*B.B;
% feat_norm=[B.B,B.B];



% feat_norm=[0.5627/(0.5627+0.5692)*A.A.A.A,0.5692/(0.5627+0.5692)*B.B.B.B];


% feat_norm=[A.A.A.A,A.A.A.A];
feat_norm=[B.B.B.B,B.B.B.B];
% feat_norm=[0.582353/(0.582353+0.582246)*A.A.A.A,0.582246/(0.582246+0.582353)*B.B.B.B];

% feat_norm=[0.5*A.A,0.5*B.B];
% feat_norm=[A.A,B.B];


% feat_norm=[A.A.A.A,A.A.A.A];
% feat_norm=[B.B.B.B,B.B.B.B];


% imgNamList=A.imgNamList;

% Ax=[0.1 -0.1];
% if feat_norm<0
%     feat_norm=0;
% else 
%     feat_norm=1;
% end
% for i=1:2
% if Ax(i)<0
%     Ax(i)=0;
% else 
%     Ax(i)=1;
% end
% end

% 
% [m,n]=size(feat_norm);
% for i=1:m
%     for j=1:n
%         if feat_norm(i,j)<0
%             feat_norm(i,j)=1;
%         else 
%             feat_norm(i,j)=0;
%         end
% %         feat_norm(i,j)=1/(1+exp(-feat_norm(i,j)));
%         disp('run...')
%     end
% end





N = 5; % 如果用于论文中，把这个�?设为你所用数据库的大�?
fid = fopen(queryFile,'rt');
queryImgs = textscan(fid, '%s');
fclose(fid);

fid = fopen(classesFile,'rt');
classesAndNum = textscan(fid, '%s %d');
fclose(fid);

for i = 1:length(classesAndNum{1, 1})
    classes{i,1} = classesAndNum{1, 1}{i,1}(1:3);
end

[numImg,d] = size(feat_norm);
querysNum = length(queryImgs{1, 1});

ap = zeros(querysNum,1);

for i =1:querysNum
    queryName = queryImgs{1, 1}{i, 1};
    queryClass = queryName(1:3);
    
    [row,col]=ind2sub(size(imgNamList),strmatch(queryName,imgNamList,'exact'));
    queryFeat = feat_norm(row, :);
    
    [row1,col1]=ind2sub(size(classesAndNum{1, 1}),strmatch(queryClass,classes,'exact'));
    queryClassNum = double(classesAndNum{1, 2}(row1,1))/1.5;
    
    %dist = distMat(queryFeat,feat_norm);
    %dist = dist';
    %[~, rank] = sort(dist, 'ascend');
    
    dist = zeros(numImg, 1);
    for j = 1:numImg
        VecTemp = feat_norm(j, :);
        dist(j) = queryFeat*VecTemp';
    end
    [~, rank] = sort(dist, 'descend');
    
    similarTerm = 0;
    
    precision = zeros(N,1);
    
    for k = 1:N
        topkClass = imgNamList{rank(k, 1), 1}(1:3);        
        if strcmp(queryClass,topkClass)==1;
            similarTerm = similarTerm+1;
            precision(k,1) = similarTerm/k;
        end
    end
    
    
    for k = 1:N
        topkClass = imgNamList{rank(k, 1), 1}(1:3); 
        % use for configure
        subplot(4,3,k);
        im = imread(imgNamList{rank(k, 1), 1});
        imshow(im);
    end
    

    ap(i,1) = sum(precision)/queryClassNum;
    
    fprintf('%s ap is %f \n',queryName,ap(i,1));
    
end

mAP = sum(ap)/querysNum;
fprintf('mAP is %f \n',mAP);




