clc;
clear all;
close all;

Img=imread('Yi.jpg','jpg');   % Original image
Img=imresize(Img,0.55);


% Img=imread('17.bmp','bmp');   % Original image
% Img=imread('lena.png','png');
% Img=imread('010_I.bmp','bmp');   % Original image

[img,KEY,xK]=main_jiami(Img);    % Encrypted algorithm


[I_yuanTu]=main_jiemi(img,KEY,xK);    % Decrypted algorithm


% % get NPCR and UACI scores
% results = NPCR_and_UACI( Img, img, 3, 255 );
% 
% 
% % get MSE and PSNR scores
% [MSE,PSNR]=PSNR_MSE(Img)   %¿¹ÔëÉù¹¥»÷ÄÜÁ¦