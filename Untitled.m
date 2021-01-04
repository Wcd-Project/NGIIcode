%加密原图
clear all;clc;close all;
I=imread('../原始、加密、解密图片/lena.png','png');  
I3=rgb2gray(I);
figure,imhist(I3);



