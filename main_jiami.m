function [img,KEY,xK]=main_jiami(I)

% clear all;
% clc;
% close all;

% I=imread('Yi.jpg','jpg');         %��ȡͼ����Ϣ
% I=imread('../ԭʼ�����ܡ�����ͼƬ/51.jpg','jpg');         %��ȡͼ����Ϣ
% I=imresize(I,0.55);

I_old=I;
I1=I(:,:,1);        %R
I2=I(:,:,2);        %G
I3=I(:,:,3);        %B

% I2=I(:,:,1);        %G
% I3=I(:,:,1);        %B


[mz,nz]=size(I1);

figure;imshow(I);
% title('ԭʼͼƬ');
figure;imhist(I1);
% title('ԭʼͼƬRͨ��ֱ��ͼ');
figure;imhist(I2);
% title('ԭʼͼƬGͨ��ֱ��ͼ');
figure;imhist(I3);
% title('ԭʼͼƬBͨ��ֱ��ͼ');
% axis([0 255 0 4000]);
[M,N]=size(I1);                      %��ͼ������и�ֵ��M,N
t=4;    %�ֿ��С

%% ԭʼͼƬR,G,Bͨ����Ϣ��
%Rͨ��
T1_R=imhist(I1);   %ͳ��ͼ��Rͨ���Ҷ�ֵ��0~255�ķֲ����������T1
S1_R=sum(T1_R);     %��������ͼ��Rͨ���ĻҶ�ֵ
xxs1_R=0;           %ԭʼͼƬRͨ�������
%Gͨ��
T1_G=imhist(I2);
S1_G=sum(T1_G);
xxs1_G=0;
%Bͨ��
T1_B=imhist(I3);
S1_B=sum(T1_B);
xxs1_B=0;

for i=1:256
    pp1_R=T1_R(i)/S1_R;   %ÿ���Ҷ�ֵռ�ȣ���ÿ���Ҷ�ֵ�ĸ���
    pp1_G=T1_G(i)/S1_G;
    pp1_B=T1_B(i)/S1_B;
    if pp1_R~=0
        xxs1_R=xxs1_R-pp1_R*log2(pp1_R);
    end
    if pp1_G~=0
        xxs1_G=xxs1_G-pp1_G*log2(pp1_G);
    end
    if pp1_B~=0
        xxs1_B=xxs1_B-pp1_B*log2(pp1_B);
    end
end

%}
NN=1000;    %���ȡ1000�����ص�
x1=ceil(rand(1,NN)*(M-1));      %����1000��1~M-1�����������Ϊ��
y1=ceil(rand(1,NN)*(N-1));      %����1000��1~N-1�����������Ϊ��
%Ԥ�����ڴ�
XX_R_SP=zeros(1,1000);YY_R_SP=zeros(1,1000);        %ˮƽ
XX_G_SP=zeros(1,1000);YY_G_SP=zeros(1,1000);
XX_B_SP=zeros(1,1000);YY_B_SP=zeros(1,1000);
XX_R_CZ=zeros(1,1000);YY_R_CZ=zeros(1,1000);        %��ֱ
XX_G_CZ=zeros(1,1000);YY_G_CZ=zeros(1,1000);
XX_B_CZ=zeros(1,1000);YY_B_CZ=zeros(1,1000);
XX_R_DJX=zeros(1,1000);YY_R_DJX=zeros(1,1000);      %�Խ���
XX_G_DJX=zeros(1,1000);YY_G_DJX=zeros(1,1000);
XX_B_DJX=zeros(1,1000);YY_B_DJX=zeros(1,1000);
for i=1:1000
    %ˮƽ
    XX_R_SP(i)=I1(x1(i),y1(i));
    YY_R_SP(i)=I1(x1(i)+1,y1(i));
    XX_G_SP(i)=I2(x1(i),y1(i));
    YY_G_SP(i)=I2(x1(i)+1,y1(i));
    XX_B_SP(i)=I3(x1(i),y1(i));
    YY_B_SP(i)=I3(x1(i)+1,y1(i));
    %��ֱ
    XX_R_CZ(i)=I1(x1(i),y1(i));
    YY_R_CZ(i)=I1(x1(i),y1(i)+1);
    XX_G_CZ(i)=I2(x1(i),y1(i));
    YY_G_CZ(i)=I2(x1(i),y1(i)+1);
    XX_B_CZ(i)=I3(x1(i),y1(i));
    YY_B_CZ(i)=I3(x1(i),y1(i)+1);
    %�Խ���
    XX_R_DJX(i)=I1(x1(i),y1(i));
    YY_R_DJX(i)=I1(x1(i)+1,y1(i)+1);
    XX_G_DJX(i)=I2(x1(i),y1(i));
    YY_G_DJX(i)=I2(x1(i)+1,y1(i)+1);
    XX_B_DJX(i)=I3(x1(i),y1(i));
    YY_B_DJX(i)=I3(x1(i)+1,y1(i)+1);
end
%ˮƽ
figure;scatter(XX_R_SP,YY_R_SP,18,'filled');
% xlabel('Rͨ����������ػҶ�ֵ');ylabel('��õ�����ˮƽ�������ػҶ�ֵ');title('ԭʼͼ��Rͨ��ˮƽ���������');
figure;scatter(XX_G_SP,YY_G_SP,18,'filled');
% xlabel('Gͨ����������ػҶ�ֵ');ylabel('��õ�����ˮƽ�������ػҶ�ֵ');title('ԭʼͼ��Gͨ��ˮƽ���������');
figure;scatter(XX_B_SP,YY_B_SP,18,'filled');
% xlabel('Bͨ����������ػҶ�ֵ');ylabel('��õ�����ˮƽ�������ػҶ�ֵ');title('ԭʼͼ��Bͨ��ˮƽ���������');
%��ֱ
figure;scatter(XX_R_CZ,YY_R_CZ,18,'filled');
% xlabel('Rͨ����������ػҶ�ֵ');ylabel('��õ����ڴ�ֱ�������ػҶ�ֵ');title('ԭʼͼ��Rͨ����ֱ���������');
figure;scatter(XX_G_CZ,YY_G_CZ,18,'filled');
% xlabel('Gͨ����������ػҶ�ֵ');ylabel('��õ����ڴ�ֱ�������ػҶ�ֵ');title('ԭʼͼ��Gͨ����ֱ���������');
figure;scatter(XX_B_CZ,YY_B_CZ,18,'filled');
% xlabel('Bͨ����������ػҶ�ֵ');ylabel('��õ����ڴ�ֱ�������ػҶ�ֵ');title('ԭʼͼ��Bͨ����ֱ���������');
%�Խ���
figure;scatter(XX_R_DJX,YY_R_DJX,18,'filled');
% xlabel('Rͨ����������ػҶ�ֵ');ylabel('��õ����ڶԽ��߷������ػҶ�ֵ');title('ԭʼͼ��Rͨ���Խ������������');
figure;scatter(XX_G_DJX,YY_G_DJX,18,'filled');
% xlabel('Gͨ����������ػҶ�ֵ');ylabel('��õ����ڶԽ��߷������ػҶ�ֵ');title('ԭʼͼ��Gͨ���Խ������������');
figure;scatter(XX_B_DJX,YY_B_DJX,18,'filled');
% xlabel('Bͨ����������ػҶ�ֵ');ylabel('��õ����ڶԽ��߷������ػҶ�ֵ');title('ԭʼͼ��Bͨ���Խ������������');
%Rͨ��
EX1_R=0;EY1_SP_R=0;DX1_R=0;DY1_SP_R=0;COVXY1_SP_R=0;    %����ˮƽ�����ʱ��Ҫ�ı���
EY1_CZ_R=0;DY1_CZ_R=0;COVXY1_CZ_R=0;                %��ֱ
EY1_DJX_R=0;DY1_DJX_R=0;COVXY1_DJX_R=0;             %�Խ���
%Gͨ��
EX1_G=0;EY1_SP_G=0;DX1_G=0;DY1_SP_G=0;COVXY1_SP_G=0;
EY1_CZ_G=0;DY1_CZ_G=0;COVXY1_CZ_G=0;
EY1_DJX_G=0;DY1_DJX_G=0;COVXY1_DJX_G=0;
%Bͨ��
EX1_B=0;EY1_SP_B=0;DX1_B=0;DY1_SP_B=0;COVXY1_SP_B=0;
EY1_CZ_B=0;DY1_CZ_B=0;COVXY1_CZ_B=0;
EY1_DJX_B=0;DY1_DJX_B=0;COVXY1_DJX_B=0;

I1=double(I1);
I2=double(I2);
I3=double(I3);
for i=1:NN
    %��һ�����ص��E��ˮƽ����ֱ���Խ���ʱ����ó��ĵ�һ�����ص��E��ͬ��ͳһ��EX1��ʾ
    EX1_R=EX1_R+I1(x1(i),y1(i)); 
    EX1_G=EX1_G+I2(x1(i),y1(i)); 
    EX1_B=EX1_B+I3(x1(i),y1(i)); 
    %�ڶ������ص��E��ˮƽ����ֱ���Խ��ߵ�E�ֱ��ӦEY1_SP��EY1_CZ��EY1_DJX
    %Rͨ��
    EY1_SP_R=EY1_SP_R+I1(x1(i),y1(i)+1);
    EY1_CZ_R=EY1_CZ_R+I1(x1(i)+1,y1(i));
    EY1_DJX_R=EY1_DJX_R+I1(x1(i)+1,y1(i)+1);
    %Gͨ��
    EY1_SP_G=EY1_SP_G+I2(x1(i),y1(i)+1);
    EY1_CZ_G=EY1_CZ_G+I2(x1(i)+1,y1(i));
    EY1_DJX_G=EY1_DJX_G+I2(x1(i)+1,y1(i)+1);
    %Bͨ��
    EY1_SP_B=EY1_SP_B+I3(x1(i),y1(i)+1);
    EY1_CZ_B=EY1_CZ_B+I3(x1(i)+1,y1(i));
    EY1_DJX_B=EY1_DJX_B+I3(x1(i)+1,y1(i)+1);
end
%ͳһ��ѭ����������ص����1000���ɼ����������
% Rͨ��
EX1_R=EX1_R/NN;
EY1_SP_R=EY1_SP_R/NN;
EY1_CZ_R=EY1_CZ_R/NN;
EY1_DJX_R=EY1_DJX_R/NN;
% Gͨ��
EX1_G=EX1_G/NN;
EY1_SP_G=EY1_SP_G/NN;
EY1_CZ_G=EY1_CZ_G/NN;
EY1_DJX_G=EY1_DJX_G/NN;
% Bͨ��
EX1_B=EX1_B/NN;
EY1_SP_B=EY1_SP_B/NN;
EY1_CZ_B=EY1_CZ_B/NN;
EY1_DJX_B=EY1_DJX_B/NN;
for i=1:NN
    %��һ�����ص��D��ˮƽ����ֱ���Խ���ʱ����ó���һ�����ص��D��ͬ��ͳһ��DX��ʾ
    DX1_R=DX1_R+(I1(x1(i),y1(i))-EX1_R)^2;
    DX1_G=DX1_G+(I2(x1(i),y1(i))-EX1_G)^2;
    DX1_B=DX1_B+(I3(x1(i),y1(i))-EX1_B)^2;
    %�ڶ������ص��D��ˮƽ����ֱ���Խ��ߵ�D�ֱ��ӦDY1_SP��DY1_CZ��DY1_DJX
    %Rͨ��
    DY1_SP_R=DY1_SP_R+(I1(x1(i),y1(i)+1)-EY1_SP_R)^2;
    DY1_CZ_R=DY1_CZ_R+(I1(x1(i)+1,y1(i))-EY1_CZ_R)^2;
    DY1_DJX_R=DY1_DJX_R+(I1(x1(i)+1,y1(i)+1)-EY1_DJX_R)^2;
    %Gͨ��
    DY1_SP_G=DY1_SP_G+(I2(x1(i),y1(i)+1)-EY1_SP_G)^2;
    DY1_CZ_G=DY1_CZ_G+(I2(x1(i)+1,y1(i))-EY1_CZ_G)^2;
    DY1_DJX_G=DY1_DJX_G+(I2(x1(i)+1,y1(i)+1)-EY1_DJX_G)^2;
    %Bͨ��
    DY1_SP_B=DY1_SP_B+(I3(x1(i),y1(i)+1)-EY1_SP_B)^2;
    DY1_CZ_B=DY1_CZ_B+(I3(x1(i)+1,y1(i))-EY1_CZ_B)^2;
    DY1_DJX_B=DY1_DJX_B+(I3(x1(i)+1,y1(i)+1)-EY1_DJX_B)^2;
    %�����������ص���غ����ļ��㣬ˮƽ����ֱ���Խ���
    %Rͨ��
    COVXY1_SP_R=COVXY1_SP_R+(I1(x1(i),y1(i))-EX1_R)*(I1(x1(i),y1(i)+1)-EY1_SP_R);
    COVXY1_CZ_R=COVXY1_CZ_R+(I1(x1(i),y1(i))-EX1_R)*(I1(x1(i)+1,y1(i))-EY1_CZ_R);
    COVXY1_DJX_R=COVXY1_DJX_R+(I1(x1(i),y1(i))-EX1_R)*(I1(x1(i)+1,y1(i)+1)-EY1_DJX_R);
    %Gͨ��
    COVXY1_SP_G=COVXY1_SP_G+(I2(x1(i),y1(i))-EX1_G)*(I2(x1(i),y1(i)+1)-EY1_SP_G);
    COVXY1_CZ_G=COVXY1_CZ_G+(I2(x1(i),y1(i))-EX1_G)*(I2(x1(i)+1,y1(i))-EY1_CZ_G);
    COVXY1_DJX_G=COVXY1_DJX_G+(I2(x1(i),y1(i))-EX1_G)*(I2(x1(i)+1,y1(i)+1)-EY1_DJX_G);
    %Bͨ��
    COVXY1_SP_B=COVXY1_SP_B+(I3(x1(i),y1(i))-EX1_B)*(I3(x1(i),y1(i)+1)-EY1_SP_B);
    COVXY1_CZ_B=COVXY1_CZ_B+(I3(x1(i),y1(i))-EX1_B)*(I3(x1(i)+1,y1(i))-EY1_CZ_B);
    COVXY1_DJX_B=COVXY1_DJX_B+(I3(x1(i),y1(i))-EX1_B)*(I3(x1(i)+1,y1(i)+1)-EY1_DJX_B);
end
%ͳһ��ѭ����������ص����1000���ɼ����������
%Rͨ��
DX1_R=DX1_R/NN;
DY1_SP_R=DY1_SP_R/NN;
DY1_CZ_R=DY1_CZ_R/NN;
DY1_DJX_R=DY1_DJX_R/NN;
COVXY1_SP_R=COVXY1_SP_R/NN;
COVXY1_CZ_R=COVXY1_CZ_R/NN;
COVXY1_DJX_R=COVXY1_DJX_R/NN;
%Gͨ��
DX1_G=DX1_G/NN;
DY1_SP_G=DY1_SP_G/NN;
DY1_CZ_G=DY1_CZ_G/NN;
DY1_DJX_G=DY1_DJX_G/NN;
COVXY1_SP_G=COVXY1_SP_G/NN;
COVXY1_CZ_G=COVXY1_CZ_G/NN;
COVXY1_DJX_G=COVXY1_DJX_G/NN;
%Bͨ��
DX1_B=DX1_B/NN;
DY1_SP_B=DY1_SP_B/NN;
DY1_CZ_B=DY1_CZ_B/NN;
DY1_DJX_B=DY1_DJX_B/NN;
COVXY1_SP_B=COVXY1_SP_B/NN;
COVXY1_CZ_B=COVXY1_CZ_B/NN;
COVXY1_DJX_B=COVXY1_DJX_B/NN;
%ˮƽ����ֱ���Խ��ߵ������
%Rͨ��
RXY1_SP_R=COVXY1_SP_R/sqrt(DX1_R*DY1_SP_R);
RXY1_CZ_R=COVXY1_CZ_R/sqrt(DX1_R*DY1_CZ_R);
RXY1_DJX_R=COVXY1_DJX_R/sqrt(DX1_R*DY1_DJX_R);
%Gͨ��
RXY1_SP_G=COVXY1_SP_G/sqrt(DX1_G*DY1_SP_G);
RXY1_CZ_G=COVXY1_CZ_G/sqrt(DX1_G*DY1_CZ_G);
RXY1_DJX_G=COVXY1_DJX_G/sqrt(DX1_G*DY1_DJX_G);
%Bͨ��
RXY1_SP_B=COVXY1_SP_B/sqrt(DX1_B*DY1_SP_B);
RXY1_CZ_B=COVXY1_CZ_B/sqrt(DX1_B*DY1_CZ_B);
RXY1_DJX_B=COVXY1_DJX_B/sqrt(DX1_B*DY1_DJX_B);

%% 1.����
%��ͼ��������������ɿ��Ա�t����������tΪ�ֿ�Ĵ�С��
M1=mod(M,t);    %����Ϊ�̶���Կ��һ�����ʱ����ȥ�����ϵ�0
N1=mod(N,t);    %����Ϊ�̶���Կ��һ�����ʱ����ȥ�����ϵ�0
if M1~=0
    I1(M+1:M+t-M1,:)=0;
    I2(M+1:M+t-M1,:)=0;
    I3(M+1:M+t-M1,:)=0;
end
if N1~=0
    I1(:,N+1:N+t-N1)=0;
    I2(:,N+1:N+t-N1)=0;
    I3(:,N+1:N+t-N1)=0;
end
[M,N]=size(I1);  %����������������
SUM=M*N;

%% 2.����Logistic��������
u=3.99;     %Logistic�����̣��Զ�Ϊ3.99
x0=sum(I1(:))/(255*SUM);     %����ó�Logistic��ֵx0
x0=floor(x0*10^4)/10^4;     %����4λС��
p=zeros(1,SUM+1000);        %Ԥ�����ڴ�
p(1)=x0;
for i=1:SUM+999                 %����SUM+999��ѭ�������õ�SUM+1000�㣨������ֵ��
    p(i+1)=u*p(i)*(1-p(i));
end
p=p(1001:length(p));            %ȥ��ǰ1000�㣬��ø��õ������

%% 3.��p���б任��0~255��Χ��������ת����M*N�Ķ�ά����R
p=mod(ceil(p*10^3),256);
R=reshape(p,N,M)';  %ת��M��N�е��������R

%% 4.���Chen�ϳ�����ϵͳ
%���ĸ���ֵX0,Y0,Z0,H0
r=(M/t)*(N/t);      %rΪ�ֿ����
%����ĸ���ֵ
X0=sum(sum(bitand(I1,3)))/(3*SUM);
Y0=sum(sum(bitand(I1,12)/4))/(3*SUM);
Z0=sum(sum(bitand(I1,48)/16))/(3*SUM);
H0=sum(sum(bitand(I1,192)/64))/(3*SUM);
%������λС��
X0=floor(X0*10^4)/10^4;
Y0=floor(Y0*10^4)/10^4;
Z0=floor(Z0*10^4)/10^4;
H0=floor(H0*10^4)/10^4;
%���ݳ�ֵ�����Chen�ϳ�����ϵͳ���õ��ĸ���������
A=xulie_output(X0,Y0,Z0,H0,r);   
X=A(:,1);
X=X(1502:length(X));        %ȥ��ǰ1501���ø��õ�����ԣ�������ϵͳ���Ӻ����������1500�㣩
Y=A(:,2);
Y=Y(1502:length(Y));
Z=A(:,3);
Z=Z(1502:length(Z));
H=A(:,4);
H=H(1502:length(H));

figure,
plot3(X,Y,Z);
xlabel('y1'),ylabel('y2'),zlabel('y3')
figure,
plot3(Y,Z,H)
xlabel('y2'),ylabel('y3'),zlabel('y4')
figure,
plot(X,H),xlabel('y1'),ylabel('y4')
figure,
plot(Y,Z),xlabel('y2'),ylabel('y3')






%% 5.DNA����
%X,Y�ֱ����I��R��DNA���뷽ʽ����8�֣�1~8
%Z�������㷽ʽ����3�֣�0~2��0��ʾ�ӣ�1��ʾ����2��ʾ���
%H��ʾDNA���뷽ʽ����8�֣�1~8
X=mod(floor(X*10^4),8)+1;
Y=mod(floor(Y*10^4),8)+1;
Z=mod(floor(Z*10^4),3);
H=mod(floor(H*10^4),8)+1;
e=N/t;  %e��ʾÿһ�п��Է�Ϊ���ٿ�

Q2=DNA_code(fenkuai(t,R,1),Y(1));
%Rͨ��
Q1_R=DNA_code(fenkuai(t,I1,1),X(1));
Q_last_R=DNA_operation(Q1_R,Q2,Z(1));
Q_R(1:t,1:t)=DNA_decode(Q_last_R,H(1));
%Gͨ��
Q1_G=DNA_code(fenkuai(t,I2,1),X(1));
Q_last_G=DNA_operation(Q1_G,Q2,Z(1));
Q_G(1:t,1:t)=DNA_decode(Q_last_G,H(1));
%Bͨ��
Q1_B=DNA_code(fenkuai(t,I3,1),X(1));
Q_last_B=DNA_operation(Q1_B,Q2,Z(1));
Q_B(1:t,1:t)=DNA_decode(Q_last_B,H(1));

for i=2:r
    Q1_R=DNA_code(fenkuai(t,I1,i),X(i));   %��ԭʼͼ��Rͨ��ÿһ���ֿ鰴X��Ӧ����Ž���DNA����
    Q1_G=DNA_code(fenkuai(t,I2,i),X(i));
    Q1_B=DNA_code(fenkuai(t,I3,i),X(i));
    
    Q2=DNA_code(fenkuai(t,R,i),Y(i));   %��R��ÿһ���ֿ鰴Y��Ӧ����Ž���DNA����
    %Rͨ��
    Q3_R=DNA_operation(Q1_R,Q2,Z(i));         %��������������õĿ鰴Z��Ӧ����Ž���DNA����
    Q4_R=DNA_operation(Q3_R,Q_last_R,Z(i));     %�������ں�ǰһ�鰴Z��Ӧ�������һ�ν������㣬��Ϊ��ɢ
    Q_last_R=Q4_R;
    %Gͨ��
    Q3_G=DNA_operation(Q1_G,Q2,Z(i));
    Q4_G=DNA_operation(Q3_G,Q_last_G,Z(i));
    Q_last_G=Q4_G;
    %Bͨ��
    Q3_B=DNA_operation(Q1_B,Q2,Z(i));
    Q4_B=DNA_operation(Q3_B,Q_last_B,Z(i));
    Q_last_B=Q4_B;
    
    xx=floor(i/e)+1;
    yy=mod(i,e);
    if yy==0
        xx=xx-1;
        yy=e;
    end
    Q_R((xx-1)*t+1:xx*t,(yy-1)*t+1:yy*t)=DNA_decode(Q4_R,H(i));    %��ÿһ��ϲ���������ͼQ
    Q_G((xx-1)*t+1:xx*t,(yy-1)*t+1:yy*t)=DNA_decode(Q4_G,H(i));
    Q_B((xx-1)*t+1:xx*t,(yy-1)*t+1:yy*t)=DNA_decode(Q4_B,H(i));
end
Q_R=uint8(Q_R);
Q_G=uint8(Q_G);
Q_B=uint8(Q_B);

%% ���ü�
xx0=sum(I2(:))/(255*SUM);     %Gͨ����ƽ���Ҷ�ֵ����Ϊ��Կ
xx0=floor(xx0*10^4)/10^4;     %����4λС��
xx1=sum(I3(:))/(255*SUM);     %Bͨ����ƽ���Ҷ�ֵ����Ϊ��Կ
xx1=floor(xx1*10^4)/10^4;     %����4λС��
ppx=zeros(1,M+1000);        %Ԥ�����ڴ�
ppy=zeros(1,N+1000); 
ppx(1)=xx0;
ppy(1)=xx1;
for i=1:M+999                 %����SUM+999��ѭ�������õ�SUM+1000�㣨������ֵ��
    ppx(i+1)=u*ppx(i)*(1-ppx(i));
end
for i=1:N+999                 %����SUM+999��ѭ�������õ�SUM+1000�㣨������ֵ��
    ppy(i+1)=u*ppy(i)*(1-ppy(i));
end
ppx=ppx(1001:length(ppx));            %ȥ��ǰ1000�㣬��ø��õ������
ppy=ppy(1001:length(ppy));

[~,Ux]=sort(ppx,'descend');
[~,Uy]=sort(ppy,'descend');

for i=1:M
    temp = Q_R(i,:);
    Q_R(i,:) = Q_R(Ux(i),:);
    Q_R(Ux(i),:) = temp;
    temp = Q_G(i,:);
    Q_G(i,:) = Q_G(Ux(i),:);
    Q_G(Ux(i),:) = temp;
    temp = Q_B(i,:);
    Q_B(i,:) = Q_B(Ux(i),:);
    Q_B(Ux(i),:) = temp;
end

for i=1:N
    temp = Q_R(:,i);
    Q_R(:,i) = Q_R(:,Uy(i));
    Q_R(:,Uy(i)) = temp;
    temp = Q_G(:,i);
    Q_G(:,i) = Q_G(:,Uy(i));
    Q_G(:,Uy(i)) = temp;
    temp = Q_B(:,i);
    Q_B(:,i) = Q_B(:,Uy(i));
    Q_B(:,Uy(i)) = temp;
end

figure;imhist(Q_R);
% title('���ܺ�Rͨ��ֱ��ͼ');
axis([0 255 0 2000]);
figure;imhist(Q_G);
% title('���ܺ�Gͨ��ֱ��ͼ');
axis([0 255 0 2000]);
figure;imhist(Q_B);
% title('���ܺ�Bͨ��ֱ��ͼ');
axis([0 255 0 2000]);
Q_jiami(:,:,1)=Q_R;
Q_jiami(:,:,2)=Q_G;
Q_jiami(:,:,3)=Q_B;
% Q=imnoise(Q,'salt & pepper',0.1);   %����10%�Ľ�������
I_new=Q_jiami;
% imwrite(Q_jiami,'���ܺ��lena.png','png');     
imwrite(Q_jiami,'���ܺ��Yi.png','png'); 
% imwrite(Q_jiami,'���ܺ��51.png','png');   
figure;imshow(Q_jiami);
% title('���ܺ�ͼƬ');

%% ���ܺ���Ϣ��
%Rͨ��
T2_R=imhist(Q_R);
S2_R=sum(T2_R);
xxs2_R=0;
%Gͨ��
T2_G=imhist(Q_G);
S2_G=sum(T2_G);
xxs2_G=0;
%Bͨ��
T2_B=imhist(Q_B);
S2_B=sum(T2_B);
xxs2_B=0;
for i=1:256
    pp2_R=T2_R(i)/S2_R;
    pp2_G=T2_R(i)/S2_G;
    pp2_B=T2_R(i)/S2_B;
    if pp2_R~=0
        xxs2_R=xxs2_R-pp2_R*log2(pp2_R);
    end
    if pp2_G~=0
        xxs2_G=xxs2_G-pp2_G*log2(pp2_G);
    end
    if pp2_B~=0
        xxs2_B=xxs2_B-pp2_B*log2(pp2_B);
    end
end

%% ����ͼ������ͼ������Է���
%}
%���������
%ˮƽ
XX_R_SP=zeros(1,1000);YY_R_SP=zeros(1,1000);  %Ԥ�����ڴ�
XX_G_SP=zeros(1,1000);YY_G_SP=zeros(1,1000);
XX_B_SP=zeros(1,1000);YY_B_SP=zeros(1,1000);
%��ֱ
XX_R_CZ=zeros(1,1000);YY_R_CZ=zeros(1,1000);  %Ԥ�����ڴ�
XX_G_CZ=zeros(1,1000);YY_G_CZ=zeros(1,1000);
XX_B_CZ=zeros(1,1000);YY_B_CZ=zeros(1,1000);
%�Խ���
XX_R_DJX=zeros(1,1000);YY_R_DJX=zeros(1,1000);  %Ԥ�����ڴ�
XX_G_DJX=zeros(1,1000);YY_G_DJX=zeros(1,1000);
XX_B_DJX=zeros(1,1000);YY_B_DJX=zeros(1,1000);
for i=1:1000
    %ˮƽ
    XX_R_SP(i)=Q_R(x1(i),y1(i));
    YY_R_SP(i)=Q_R(x1(i)+1,y1(i));
    XX_G_SP(i)=Q_G(x1(i),y1(i));
    YY_G_SP(i)=Q_G(x1(i)+1,y1(i));
    XX_B_SP(i)=Q_B(x1(i),y1(i));
    YY_B_SP(i)=Q_B(x1(i)+1,y1(i));
    %��ֱ
    XX_R_CZ(i)=Q_R(x1(i),y1(i));
    YY_R_CZ(i)=Q_R(x1(i),y1(i)+1);
    XX_G_CZ(i)=Q_G(x1(i),y1(i));
    YY_G_CZ(i)=Q_G(x1(i),y1(i)+1);
    XX_B_CZ(i)=Q_B(x1(i),y1(i));
    YY_B_CZ(i)=Q_B(x1(i),y1(i)+1);
    %�Խ���
    XX_R_DJX(i)=Q_R(x1(i),y1(i));
    YY_R_DJX(i)=Q_R(x1(i)+1,y1(i)+1);
    XX_G_DJX(i)=Q_G(x1(i),y1(i));
    YY_G_DJX(i)=Q_G(x1(i)+1,y1(i)+1);
    XX_B_DJX(i)=Q_B(x1(i),y1(i));
    YY_B_DJX(i)=Q_B(x1(i)+1,y1(i)+1);
end
%ˮƽ
figure;scatter(XX_R_SP,YY_R_SP,18,'filled');
% xlabel('Rͨ����������ػҶ�ֵ');ylabel('��õ�����ˮƽ�������ػҶ�ֵ');title('����ͼ��Rͨ��ˮƽ���������');
figure;scatter(XX_G_SP,YY_G_SP,18,'filled');
% xlabel('Gͨ����������ػҶ�ֵ');ylabel('��õ�����ˮƽ�������ػҶ�ֵ');title('����ͼ��Gͨ��ˮƽ���������');
figure;scatter(XX_B_SP,YY_B_SP,18,'filled');
% xlabel('Bͨ����������ػҶ�ֵ');ylabel('��õ�����ˮƽ�������ػҶ�ֵ');title('����ͼ��Bͨ��ˮƽ���������');
%��ֱ
figure;scatter(XX_R_CZ,YY_R_CZ,18,'filled');
% xlabel('Rͨ����������ػҶ�ֵ');ylabel('��õ����ڴ�ֱ�������ػҶ�ֵ');title('����ͼ��Rͨ����ֱ���������');
figure;scatter(XX_G_CZ,YY_G_CZ,18,'filled');
% xlabel('Gͨ����������ػҶ�ֵ');ylabel('��õ����ڴ�ֱ�������ػҶ�ֵ');title('����ͼ��Gͨ����ֱ���������');
figure;scatter(XX_B_CZ,YY_B_CZ,18,'filled');
% xlabel('Bͨ����������ػҶ�ֵ');ylabel('��õ����ڴ�ֱ�������ػҶ�ֵ');title('����ͼ��Bͨ����ֱ���������');
%�Խ���
figure;scatter(XX_R_DJX,YY_R_DJX,18,'filled');
% xlabel('Rͨ����������ػҶ�ֵ');ylabel('��õ����ڶԽ��߷������ػҶ�ֵ');title('����ͼ��Rͨ���Խ������������');
figure;scatter(XX_G_DJX,YY_G_DJX,18,'filled');
% xlabel('Gͨ����������ػҶ�ֵ');ylabel('��õ����ڶԽ��߷������ػҶ�ֵ');title('����ͼ��Gͨ���Խ������������');
figure;scatter(XX_B_DJX,YY_B_DJX,18,'filled');
% xlabel('Bͨ����������ػҶ�ֵ');ylabel('��õ����ڶԽ��߷������ػҶ�ֵ');title('����ͼ��Bͨ���Խ������������');
%Rͨ��
Q_R=double(Q_R);
EX2_R=0;EY2_SP_R=0;DX2_R=0;DY2_SP_R=0;COVXY2_SP_R=0;    %ˮƽ
EY2_CZ_R=0;DY2_CZ_R=0;COVXY2_CZ_R=0;    %��ֱ
EY2_DJX_R=0;DY2_DJX_R=0;COVXY2_DJX_R=0;   %�Խ���
%Gͨ��
Q_G=double(Q_G);
EX2_G=0;EY2_SP_G=0;DX2_G=0;DY2_SP_G=0;COVXY2_SP_G=0;    %ˮƽ
EY2_CZ_G=0;DY2_CZ_G=0;COVXY2_CZ_G=0;    %��ֱ
EY2_DJX_G=0;DY2_DJX_G=0;COVXY2_DJX_G=0;   %�Խ���
%Bͨ��
Q_B=double(Q_B);
EX2_B=0;EY2_SP_B=0;DX2_B=0;DY2_SP_B=0;COVXY2_SP_B=0;    %ˮƽ
EY2_CZ_B=0;DY2_CZ_B=0;COVXY2_CZ_B=0;    %��ֱ
EY2_DJX_B=0;DY2_DJX_B=0;COVXY2_DJX_B=0;   %�Խ���
for i=1:NN
    %��һ�����ص��E��ˮƽ����ֱ���Խ���ʱ����ó��ĵ�һ�����ص��E��ͬ��ͳһ��EX2��ʾ
    EX2_R=EX2_R+Q_R(x1(i),y1(i));
    EX2_G=EX2_G+Q_G(x1(i),y1(i));
    EX2_B=EX2_B+Q_B(x1(i),y1(i));
    %�ڶ������ص��E��ˮƽ����ֱ���Խ��ߵ�E�ֱ��ӦEY2_SP��EY2_CZ��EY2_DJX
    %Rͨ��
    EY2_SP_R=EY2_SP_R+Q_R(x1(i),y1(i)+1);
    EY2_CZ_R=EY2_CZ_R+Q_R(x1(i)+1,y1(i));
    EY2_DJX_R=EY2_DJX_R+Q_R(x1(i)+1,y1(i)+1);
    %Gͨ��
    EY2_SP_G=EY2_SP_G+Q_G(x1(i),y1(i)+1);
    EY2_CZ_G=EY2_CZ_G+Q_G(x1(i)+1,y1(i));
    EY2_DJX_G=EY2_DJX_G+Q_G(x1(i)+1,y1(i)+1);
    %Bͨ��
    EY2_SP_B=EY2_SP_B+Q_B(x1(i),y1(i)+1);
    EY2_CZ_B=EY2_CZ_B+Q_B(x1(i)+1,y1(i));
    EY2_DJX_B=EY2_DJX_B+Q_B(x1(i)+1,y1(i)+1);
end
%ͳһ��ѭ����������ص����1000���ɼ����������
%Rͨ��
EX2_R=EX2_R/NN;
EY2_SP_R=EY2_SP_R/NN;
EY2_CZ_R=EY2_CZ_R/NN;
EY2_DJX_R=EY2_DJX_R/NN;
%Gͨ��
EX2_G=EX2_G/NN;
EY2_SP_G=EY2_SP_G/NN;
EY2_CZ_G=EY2_CZ_G/NN;
EY2_DJX_G=EY2_DJX_G/NN;
%Bͨ��
EX2_B=EX2_B/NN;
EY2_SP_B=EY2_SP_B/NN;
EY2_CZ_B=EY2_CZ_B/NN;
EY2_DJX_B=EY2_DJX_B/NN;

for i=1:NN
    %��һ�����ص��D��ˮƽ����ֱ���Խ���ʱ����ó���һ�����ص��D��ͬ��ͳһ��DX2��ʾ
    DX2_R=DX2_R+(Q_R(x1(i),y1(i))-EX2_R)^2;
    DX2_G=DX2_G+(Q_G(x1(i),y1(i))-EX2_G)^2;
    DX2_B=DX2_B+(Q_B(x1(i),y1(i))-EX2_B)^2;
    %�ڶ������ص��D��ˮƽ����ֱ���Խ��ߵ�D�ֱ��ӦDY2_SP��DY2_CZ��DY2_DJX
    %Rͨ��
    DY2_SP_R=DY2_SP_R+(Q_R(x1(i),y1(i)+1)-EY2_SP_R)^2;
    DY2_CZ_R=DY2_CZ_R+(Q_R(x1(i)+1,y1(i))-EY2_CZ_R)^2;
    DY2_DJX_R=DY2_DJX_R+(Q_R(x1(i)+1,y1(i)+1)-EY2_DJX_R)^2;
    %Gͨ��
    DY2_SP_G=DY2_SP_G+(Q_G(x1(i),y1(i)+1)-EY2_SP_G)^2;
    DY2_CZ_G=DY2_CZ_G+(Q_G(x1(i)+1,y1(i))-EY2_CZ_G)^2;
    DY2_DJX_G=DY2_DJX_G+(Q_G(x1(i)+1,y1(i)+1)-EY2_DJX_G)^2;
    %Bͨ��
    DY2_SP_B=DY2_SP_B+(Q_B(x1(i),y1(i)+1)-EY2_SP_B)^2;
    DY2_CZ_B=DY2_CZ_B+(Q_B(x1(i)+1,y1(i))-EY2_CZ_B)^2;
    DY2_DJX_B=DY2_DJX_B+(Q_B(x1(i)+1,y1(i)+1)-EY2_DJX_B)^2;
    %�����������ص���غ����ļ��㣬ˮƽ����ֱ���Խ���
    %Rͨ��
    COVXY2_SP_R=COVXY2_SP_R+(Q_R(x1(i),y1(i))-EX2_R)*(Q_R(x1(i),y1(i)+1)-EY2_SP_R);
    COVXY2_CZ_R=COVXY2_CZ_R+(Q_R(x1(i),y1(i))-EX2_R)*(Q_R(x1(i)+1,y1(i))-EY2_CZ_R);
    COVXY2_DJX_R=COVXY2_DJX_R+(Q_R(x1(i),y1(i))-EX2_R)*(Q_R(x1(i)+1,y1(i)+1)-EY2_DJX_R);
    %Gͨ��
    COVXY2_SP_G=COVXY2_SP_G+(Q_G(x1(i),y1(i))-EX2_G)*(Q_G(x1(i),y1(i)+1)-EY2_SP_G);
    COVXY2_CZ_G=COVXY2_CZ_G+(Q_G(x1(i),y1(i))-EX2_G)*(Q_G(x1(i)+1,y1(i))-EY2_CZ_G);
    COVXY2_DJX_G=COVXY2_DJX_G+(Q_G(x1(i),y1(i))-EX2_G)*(Q_G(x1(i)+1,y1(i)+1)-EY2_DJX_G);
    %Bͨ��
    COVXY2_SP_B=COVXY2_SP_B+(Q_B(x1(i),y1(i))-EX2_B)*(Q_B(x1(i),y1(i)+1)-EY2_SP_B);
    COVXY2_CZ_B=COVXY2_CZ_B+(Q_B(x1(i),y1(i))-EX2_B)*(Q_B(x1(i)+1,y1(i))-EY2_CZ_B);
    COVXY2_DJX_B=COVXY2_DJX_B+(Q_B(x1(i),y1(i))-EX2_B)*(Q_B(x1(i)+1,y1(i)+1)-EY2_DJX_B);
end
%ͳһ��ѭ����������ص����1000���ɼ����������
%Rͨ��
DX2_R=DX2_R/NN;
DY2_SP_R=DY2_SP_R/NN;
DY2_CZ_R=DY2_CZ_R/NN;
DY2_DJX_R=DY2_DJX_R/NN;
COVXY2_SP_R=COVXY2_SP_R/NN;
COVXY2_CZ_R=COVXY2_CZ_R/NN;
COVXY2_DJX_R=COVXY2_DJX_R/NN;
%Gͨ��
DX2_G=DX2_G/NN;
DY2_SP_G=DY2_SP_G/NN;
DY2_CZ_G=DY2_CZ_G/NN;
DY2_DJX_G=DY2_DJX_G/NN;
COVXY2_SP_G=COVXY2_SP_G/NN;
COVXY2_CZ_G=COVXY2_CZ_G/NN;
COVXY2_DJX_G=COVXY2_DJX_G/NN;
%Bͨ��
DX2_B=DX2_B/NN;
DY2_SP_B=DY2_SP_B/NN;
DY2_CZ_B=DY2_CZ_B/NN;
DY2_DJX_B=DY2_DJX_B/NN;
COVXY2_SP_B=COVXY2_SP_B/NN;
COVXY2_CZ_B=COVXY2_CZ_B/NN;
COVXY2_DJX_B=COVXY2_DJX_B/NN;
%ˮƽ����ֱ���Խ��ߵ������
%Rͨ��
RXY2_SP_R=COVXY2_SP_R/sqrt(DX2_R*DY2_SP_R);
RXY2_CZ_R=COVXY2_CZ_R/sqrt(DX2_R*DY2_CZ_R);
RXY2_DJX_R=COVXY2_DJX_R/sqrt(DX2_R*DY2_DJX_R);
%Gͨ��
RXY2_SP_G=COVXY2_SP_G/sqrt(DX2_G*DY2_SP_G);
RXY2_CZ_G=COVXY2_CZ_G/sqrt(DX2_G*DY2_CZ_G);
RXY2_DJX_G=COVXY2_DJX_G/sqrt(DX2_G*DY2_DJX_G);
%Bͨ��
RXY2_SP_B=COVXY2_SP_B/sqrt(DX2_B*DY2_SP_B);
RXY2_CZ_B=COVXY2_CZ_B/sqrt(DX2_B*DY2_CZ_B);
RXY2_DJX_B=COVXY2_DJX_B/sqrt(DX2_B*DY2_DJX_B);
% close all;
%% ���������Ϣ
disp('���ܳɹ�');
disp('��Կ��');
disp(['��Կ1����=',num2str(u),'     ��Կ2��x0=',num2str(x0),'    ��Կ3��x(0)=',num2str(X0),'    ��Կ4��y(0)=',num2str(Y0),'   ��Կ5��z(0)=',num2str(Z0),]);
disp(['��Կ6��h(0)=',num2str(H0),'   ��Կ7��M1=',num2str(M1),'   ��Կ8��N1=',num2str(N1),'   ��Կ9��xx0=',num2str(xx0),'   ��Կ10��xx1=',num2str(xx1)]);
disp('��Ϣ�أ�');
disp(['ԭʼͼƬRͨ����Ϣ��=',num2str(xxs1_R),'  ԭʼͼƬGͨ����Ϣ��=',num2str(xxs1_G),'  ԭʼͼƬBͨ����Ϣ��=',num2str(xxs1_B)]);
disp(['����ͼƬRͨ����Ϣ��=',num2str(xxs2_R),'  ����ͼƬGͨ����Ϣ��=',num2str(xxs2_G),'  ����ͼƬBͨ����Ϣ��=',num2str(xxs2_B)]);
disp('Rͨ������ԣ�');
disp(['ԭʼͼƬRͨ������ԣ�','  ˮƽ�����=',num2str(RXY1_SP_R),'    ��ֱ�����=',num2str(RXY1_CZ_R),'  �Խ��������=',num2str(RXY1_DJX_R)]);
disp(['����ͼƬRͨ������ԣ�','  ˮƽ�����=',num2str(RXY2_SP_R),'  ��ֱ�����=',num2str(RXY2_CZ_R),'  �Խ��������=',num2str(RXY2_DJX_R)]);
disp('Gͨ������ԣ�');
disp(['ԭʼͼƬGͨ������ԣ�','  ˮƽ�����=',num2str(RXY1_SP_G),'   ��ֱ�����=',num2str(RXY1_CZ_G),'  �Խ��������=',num2str(RXY1_DJX_G)]);
disp(['����ͼƬGͨ������ԣ�','  ˮƽ�����=',num2str(RXY2_SP_G),'  ��ֱ�����=',num2str(RXY2_CZ_G),'  �Խ��������=',num2str(RXY2_DJX_G)]);
disp('Bͨ������ԣ�');
disp(['ԭʼͼƬBͨ������ԣ�','  ˮƽ�����=',num2str(RXY1_SP_B),'   ��ֱ�����=',num2str(RXY1_CZ_B),'  �Խ��������=',num2str(RXY1_DJX_B)]);
disp(['����ͼƬBͨ������ԣ�','  ˮƽ�����=',num2str(RXY2_SP_B),'  ��ֱ�����=',num2str(RXY2_CZ_B),'  �Խ��������=',num2str(RXY2_DJX_B)]);

key1=u;
key2=x0;
key3=X0;
key4=Y0;
key5=Z0;
key6=H0;
xK0=xx0;
xK1=xx1;
KEY=[key1,key2,key3,key4,key5,key6];
img=[I_new];
xK=[xK0,xK1];
end
