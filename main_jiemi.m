function [I_yuanTu]=main_jiemi(img,KEY,xK)
% clear all;
% clc;
% close all;

% I=imread('加密后的Yi.png','png');   


% I=imread('加密后的lena.png','png');           %读取图像信息
% I=imread('加密后的51.png','png');   
% I=imresize(I,0.5);

I=img;
% 
% % I=imnoise(I, 'gaussian', 0, 0.0005);
% 
% 
I1=I(:,:,1);     %R通道
I2=I(:,:,2);     %G通道
I3=I(:,:,3);     %B通道
% % imhist(I2);
[m1,n1]=size(I1);
% % % I11([1:m1],[1:n1])=0;
% % % % 
% % 
[m2,n2]=size(I2);
% % % I12([1:m2],[1:n2])=0;
% % % % 
% % 
[m3,n3]=size(I3);
% % I13([1:m3],[1:n3])=0;
% % 
% 
%  I11([m1/2:m1],[n1/2:n1])=I1([m1/2:m1],[n1/2:n1]);
%  I12([m2/2:m2],[n2/2:n2])=I2([m2/2:m2],[n2/2:n2]);
%  I13([m3/2:m3],[n3/2:n3])=I3([m3/2:m3],[n3/2:n3]);
% %  
%  I11([m1/2:m1],[1:n1])=I1([m1/2:m1],[1:n1]);
%  I12([m2/2:m2],[1:n2])=I2([m2/2:m2],[1:n2]);
%  I13([m3/2:m3],[1:n3])=I3([m3/2:m3],[1:n3]);
%  % % 
%  I1([1:m1/2],[1:n1/2])=0; %1/4
%  I2([1:m2/2],[1:n2/2])=0;
%  I3([1:m3/2],[1:n3/2])=0;
% % % % 
%  I1([1:m1/4],[1:n1])=0; %1/4
%  I2([1:m2/4],[1:n2])=0;
%  I3([1:m3/4],[1:n3])=0;
% % % % 
%  I1([1:m1],[1:n1/4])=0; %1/4
%  I2([1:m2],[1:n2/4])=0;
%  I3([1:m3],[1:n3/4])=0;
% % 


%  I1([1:m1/2],[1:n1/2])=0; %1/2
%  I2([1:m2/2],[1:n2/2])=0;
%  I3([1:m3/2],[1:n3/2])=0;
 
 
I=cat(3,I1,I2,I3);

figure,imshow(I);






% % [m,n,~]=size(I);
% %  
% % %采样数据总数
% % amount = fix(m*n*1/4*1);
% % sampledata = zeros(amount,3);
% % %随机生成采样点的坐标
% % for j=1:amount
% %     x = randi(m,1,1);    
% %     y = randi(n,1,1);        
% %     %取得的数据
% %     sampledata(j,:) = I(x,y,:);
% %     %在图上标注为白色
% %     I(x,y,:)=[255 255 255];
% % end
% % figure,imshow(I);












I1=I(:,:,1);     %R通道
I2=I(:,:,2);     %G通道
I3=I(:,:,3);     %B通道
[M,N]=size(I1);                      %将图像的行列赋值给M,N
t=4;    %分块大小
M1=0;   %加密时补零的参数，M1=mod(M,t);作为密钥
N1=0;   %加密时补零的参数，N1=mod(N,t);作为密钥
SUM=M*N;
u=KEY(1);     %密钥1：μ
% xx0=0.5008;
% xx1=0.5109;
% xx0=0.3883;
% xx1=0.4134;
% % % % % xx0=0.633;
% % % % % xx1=0.9043;
xx0=xK(1);
xx1=xK(2);


% xx0=0.586;
% xx1=0.8473;


ppx=zeros(1,M+1000);        %预分配内存
ppy=zeros(1,N+1000); 
ppx(1)=xx0;
ppy(1)=xx1;
for i=1:M+999                 %进行SUM+999次循环，共得到SUM+1000点（包括初值）
    ppx(i+1)=u*ppx(i)*(1-ppx(i));
end
for i=1:N+999                 %进行SUM+999次循环，共得到SUM+1000点（包括初值）
    ppy(i+1)=u*ppy(i)*(1-ppy(i));
end
ppx=ppx(1001:length(ppx));            %去除前1000点，获得更好的随机性
ppy=ppy(1001:length(ppy));

[~,Ux]=sort(ppx,'descend');
[~,Uy]=sort(ppy,'descend');

for i=N:-1:1
    temp = I1(:,i);
    I1(:,i) = I1(:,Uy(i));
    I1(:,Uy(i)) = temp;
    temp = I2(:,i);
    I2(:,i) = I2(:,Uy(i));
    I2(:,Uy(i)) = temp;
    temp = I3(:,i);
    I3(:,i) = I3(:,Uy(i));
    I3(:,Uy(i)) = temp;
end
for i=M:-1:1
    temp = I1(i,:);
    I1(i,:) = I1(Ux(i),:);
    I1(Ux(i),:) = temp;
    temp = I2(i,:);
    I2(i,:) = I2(Ux(i),:);
    I2(Ux(i),:) = temp;
    temp = I3(i,:);
    I3(i,:) = I3(Ux(i),:);
    I3(Ux(i),:) = temp;
end
%% 2.产生Logistic混沌序列
% u=3.990000000000001; %密钥敏感性测试  10^-15
%u=3.99;%密钥：Logistic参数μ
% x0=0.7067000000000001; %密钥敏感性测试  10^-16
% x0=0.7067; %密钥：Logistic初值x0
% % % x0=0.9355; %密钥：Logistic初值x0
x0=KEY(2);
% x0=0.7814; %密钥：Logistic初值x0
% x0=0.3462;            %home图片
p=zeros(1,SUM+1000);
p(1)=x0;
for i=1:SUM+999                        %进行N-1次循环
    p(i+1)=u*p(i)*(1-p(i));          %循环产生密码
end
p=p(1001:length(p));

%% 3.将p序列变换到0~255范围内整数，转换成M*N的二维矩阵R
p=mod(ceil(p*10^3),256);
R=reshape(p,N,M)';  %转成M行N列

%% 4.求解混沌方程
%求四个初值X0,Y0,Z0,H0
r=(M/t)*(N/t);
% X0=0.5008000000000001;        %密钥敏感性测试
% X0=0.5008;
% Y0=0.5109;
% Z0=0.4893;
% H0=0.7765;
% X0=0.5008;
% Y0=0.5109;
% Z0=0.4893;
% H0=0.7765;
% X0=0.5056;        %home图片
% Y0=0.505;
% Z0=0.4564;
% H0=0.3062;

% X0=0.4999;        %Yi
% Y0=0.4991;
% Z0=0.478;
% H0=0.8793;

% % X0=0.5019;        %Yi
% % Y0=0.4888;
% % Z0=0.8209;
% % H0=0.9988;

X0=KEY(3);
Y0=KEY(4);
Z0=KEY(5);
H0=KEY(6);

A=xulie_output(X0,Y0,Z0,H0,r);
X=A(:,1);
X=X(1502:length(X));
Y=A(:,2);
Y=Y(1502:length(Y));
Z=A(:,3);
Z=Z(1502:length(Z));
H=A(:,4);
H=H(1502:length(H));

%% 5.DNA编码
%X,Y分别决定I和R的DNA编码方式，有8种，1~8
X=mod(floor(X*10^4),8)+1;
Y=mod(floor(Y*10^4),8)+1;
Z=mod(floor(Z*10^4),3);
Z(Z==0)=3;
Z(Z==1)=0;
Z(Z==3)=1;
H=mod(floor(H*10^4),8)+1;
e=N/t;
for i=r:-1:2
    Q1_R=DNA_code(fenkuai(t,I1,i),H(i));
    Q1_G=DNA_code(fenkuai(t,I2,i),H(i));
    Q1_B=DNA_code(fenkuai(t,I3,i),H(i));
    
    Q1_last_R=DNA_code(fenkuai(t,I1,i-1),H(i-1));
    Q1_last_G=DNA_code(fenkuai(t,I2,i-1),H(i-1));
    Q1_last_B=DNA_code(fenkuai(t,I3,i-1),H(i-1));
    
    Q2_R=DNA_operation(Q1_R,Q1_last_R,Z(i));        %扩散前
    Q2_G=DNA_operation(Q1_G,Q1_last_G,Z(i));
    Q2_B=DNA_operation(Q1_B,Q1_last_B,Z(i));

    Q3=DNA_code(fenkuai(t,R,i),Y(i));
    
    Q4_R=DNA_operation(Q2_R,Q3,Z(i));
    Q4_G=DNA_operation(Q2_G,Q3,Z(i));
    Q4_B=DNA_operation(Q2_B,Q3,Z(i));
    
    xx=floor(i/e)+1;
    yy=mod(i,e);
    if yy==0
        xx=xx-1;
        yy=e;
    end
    I1((xx-1)*t+1:xx*t,(yy-1)*t+1:yy*t)=DNA_decode(Q4_R,X(i));
    I2((xx-1)*t+1:xx*t,(yy-1)*t+1:yy*t)=DNA_decode(Q4_G,X(i));
    I3((xx-1)*t+1:xx*t,(yy-1)*t+1:yy*t)=DNA_decode(Q4_B,X(i));
end
Q5_R=DNA_code(fenkuai(t,I1,1),H(1));
Q5_G=DNA_code(fenkuai(t,I2,1),H(1));
Q5_B=DNA_code(fenkuai(t,I3,1),H(1));

Q6=DNA_code(fenkuai(t,R,1),Y(1));

Q7_R=DNA_operation(Q5_R,Q6,Z(1));
Q7_G=DNA_operation(Q5_G,Q6,Z(1));
Q7_B=DNA_operation(Q5_B,Q6,Z(1));

I1(1:t,1:t)=DNA_decode(Q7_R,X(1));
I2(1:t,1:t)=DNA_decode(Q7_G,X(1));
I3(1:t,1:t)=DNA_decode(Q7_B,X(1));

Q_jiemi(:,:,1)=uint8(I1);
Q_jiemi(:,:,2)=uint8(I2);
Q_jiemi(:,:,3)=uint8(I3);

%% 6、去除加密时补的零
if M1~=0
    Q_jiemi=Q_jiemi(1:M-t+M1,:,:);
end
if N1~=0
    Q_jiemi=Q_jiemi(:,1:N-t+N1,:);
end
%比较解密后的图与原图是否完全相同
% II=imread('../原始、加密、解密图片/Yi1.jpg','jpg');
% cha=sum(sum(sum(Q_jiemi-II)));      %两幅图做差后求总和
%% 保存图片
% imwrite(Q_jiemi,'解密后的lena.png','png');    
% imwrite(Q_jiemi,'解密后的Yi.jpg','jpg'); 
I_yuanTu=Q_jiemi;
imwrite(Q_jiemi,'解密后的Yi.png','png'); 
disp('您输入的解密密钥为：');
disp(['密钥1：μ=',num2str(u),'     密钥2：x0=',num2str(x0),'    密钥3：x(0)=',num2str(X0),'    密钥4：y(0)=',num2str(Y0),'   密钥5：z(0)=',num2str(Z0),]);
disp(['密钥6：h(0)=',num2str(H0),'   密钥7：M1=',num2str(M1),'   密钥8：N1=',num2str(N1),'   密钥9：xx0=',num2str(xx0),'   密钥10：xx1=',num2str(xx1)]);
disp('解密完成'); 
figure,imshow(Q_jiemi);
% title('解密后图片');