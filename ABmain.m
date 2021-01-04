clc;clear all;close all;
% A=load('feat4096Norml-16RGB-512000.mat');
% B=load('feat4096Norml-19RGB-512000.mat');
% A=A.feat_norm;
% B=B.feat_norm;
% AB=A1.*B1;
A=load('A_FDXJ.mat');
B=load('B_FDXJ.mat');
% feat_norm=0.01*A.A+20*B.B;
% feat_norm=[B.B,B.B];
AB=[0.5627/(0.5627+0.5692)*A.A.A,0.5692/(0.5627+0.5692)*B.B.B];
% feat_norm=[0.5*A.A,0.5*B.B];
% feat_norm=[A.A,B.B];
% feat_norm=[A.A,A.A];


% save('AB_FDXJ.mat');
% AB.feat_norm=AB;
% AB.imgNamList=A.imgNamList;
imgNamList=A.imgNamList;
save('AB_FDXJ.mat','AB','imgNamList', '-v7.3');
% B=B.feat_norm;
% A=A.feat_norm;
save('A_FDXJ.mat','A','imgNamList', '-v7.3');
save('B_FDXJ.mat','B','imgNamList', '-v7.3');

