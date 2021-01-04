% Author: Yong Yuan
% Homepage: yongyuan.name
% If matconvnet-1.0-beta18 is used, and get the error "Reference to
% non-existent field 'precious'". You must download the lastest pre-trained model
% https://github.com/vlfeat/matconvnet/issues/389

clear all;close all;clc;

% version: matconvnet-1.0-beta17
% run C:\Users\Administrator\Desktop\CNN-for-Image-Retrieval-master\matconvnet-1.0-beta17\matlab\vl_compilenn
run matconvnet-1.0-beta17\matlab\vl_setupnn

%% Step 1 lOADING PATHS

path_imgDB = 'database/';


addpath(path_imgDB);
addpath tools;

% viesion: matconvnet-1.0-beta17
net = load('imagenet-vgg-f.mat') ;
% net = load('C:\Users\Administrator\Desktop\CBIR-master\CNN-for-Image-Retrieval-master\imagenet-vgg-s.mat') ;
% net = load('C:\Users\Administrator\Desktop\CBIR-master\CNN-for-Image-Retrieval-master\imagenet-vgg-m.mat') ;
% net = load('C:\Users\Administrator\Desktop\CBIR-master\CNN-for-Image-Retrieval-master\imagenet-vgg-verydeep-16.mat') ;
% net = load('C:\Users\Administrator\Desktop\CBIR-master\CNN-for-Image-Retrieval-master\imagenet-vgg-verydeep-19.mat') ;
% net = load('C:\Users\Administrator\Desktop\CBIR-master\CNN-for-Image-Retrieval-master\imagenet-resnet-50-dag.mat') ;
% net = resnet50();
% layers =net1.layers;
% meta=net1.meta;

% XX=randn(1,253);
% % XX=(ones(1,253))/253;
% net.layers{1,20}.weight{1,2}=XX';
% net.layers{1,20}.size=[1,1,4096,253];



%% Step 2 LOADING IMAGE AND EXTRACTING FEATURE
imgFiles = dir(path_imgDB);
imgNamList = {imgFiles(~[imgFiles.isdir]).name};
clear imgFiles;
imgNamList = imgNamList';

numImg = length(imgNamList);
feat = [];
rgbImgList = {};

%parpool;

%parfor i = 1:numImg
for i = 1:numImg
   oriImg = imread(imgNamList{i, 1}); 
   if size(oriImg, 3) == 3
       im_ = single(oriImg) ; % note: 255 range
       im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
       
%        im_ = im_ - net.meta.normalization.averageImage ;
% net.meta.normalization.averageImage(:,:,1)=mean2(net.meta.normalization.averageImage(:,:,1));
% net.meta.normalization.averageImage(:,:,2)=mean2(net.meta.normalization.averageImage(:,:,2));
% net.meta.normalization.averageImage(:,:,3)=mean2(net.meta.normalization.averageImage(:,:,3));
       im_ = im_ - net.meta.normalization.averageImage ;
       
       res = vl_simplenn(net, im_) ;
       
       % viesion: matconvnet-1.0-beta17
       featVec = res(20).x;
%             featVec = res(36).x;  
%                    featVec = res(42).x;  

%                     featVec = (net.params(208).value)';  

       featVec = featVec(:);
       
       feat = [feat; featVec'];
       fprintf('extract %d image\n\n', i);
   else
       im_ = single(repmat(oriImg,[1 1 3])) ; % note: 255 range
       im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
       im_ = im_ - net.meta.normalization.averageImage ;
       res = vl_simplenn(net, im_) ;


%        im_ = single(repmat(oriImg,[1 1 3])) ; % note: 255 range
%        im_ = imresize(im_, meta.normalization.imageSize(1:2)) ;
%        im_ = im_ - meta.normalization.averageImage ;
% %        net={layers;meta};
%         net.layers=struct2cell(layers);
%         net.meta=meta;
%        res = vl_simplenn(net, im_) ;
       
       
       
       
       
       % viesion: matconvnet-1.0-beta17
       featVec = res(20).x;
%            featVec = res(36).x;   
%                    featVec = res(42).x;  

%  featVec = (params(208).value)';      
       featVec = featVec(:);
       feat = [feat; featVec'];
       fprintf('extract %d image\n\n', i);
   end
end

% reduce demension by PCA, recomend to reduce it to 128 dimension.
% ½µÎ¬
% [coeff, score, latent] = princomp(feat);
[coeff, score, latent] = pca(feat);
% feat = feat*coeff(:, 1:512);
feat = feat*coeff(:, 1:512);

feat_norm = normalize1(feat);
    
    
save('feat4096Norml-ftest.mat','feat_norm', 'imgNamList', '-v7.3');
% save('netlayers.mat','netlayers', '-v7.3');



