clear all;close all;clc;
% I=imread('../ԭʼ�����ܡ�����ͼƬ/Yi.jpg','jpg');  
I=imread('../ԭʼ�����ܡ�����ͼƬ/���ܺ��Yi.png','png');  
I=imresize(I,0.5);
[m,n,~]=size(I);
 
%������������
amount = fix(m*n*0.1);
sampledata = zeros(amount,3);
%������ɲ����������
for j=1:amount
    x = randi(m,1,1);    
    y = randi(n,1,1);        
    %ȡ�õ�����
    sampledata(j,:) = I(x,y,:);
    %��ͼ�ϱ�עΪ��ɫ
    I(x,y,:)=[255 255 255];
end
%��ʾ������
imshow(I);