       % Demo 1: simple use
       % generate two 256x256 8-bit random-imges
%        img_a = randi(256,256,256) - 1; 
%        img_b = randi(256,256,256) - 1; 
       
       clc;close all;clear all;
       img_a=imread('../原始、加密、解密图片/Yi.jpg','jpg');         %读取图像信息
       img_b=imread('../原始、加密、解密图片/加密后的Yi.png','png');   
       % get NPCR and UACI scores
       results = NPCR_and_UACI( img_a, img_b, 3, 255 );