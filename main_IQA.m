%% ͼ���������ۡ���������Mean Square Error,MSE������ֵ����ȣ�Peak-Signal to Noise Ratio,PSNR��
clear;clc;
I=imread('../ԭʼ�����ܡ�����ͼƬ/���ܺ��lena.png','png');           %��ȡͼ����Ϣ
[M,N]=size(I(:,:,1));                      %��ͼ������и�ֵ��M,N
t=4;    %�ֿ��С
M1=0;   %����ʱ����Ĳ�����M1=mod(M,t);��Ϊ��Կ
N1=0;   %����ʱ����Ĳ�����N1=mod(N,t);��Ϊ��Կ
SUM=M*N;
% I=imnoise(I,'salt & pepper',0.1);         %����10%�Ľ�������
%�����˹����������ͼ����������

%% ����Logistic��������
u=3.99;%Logistic������
x0=0.7067; %Logistic��ֵx0
p=zeros(1,SUM+1000);
p(1)=x0;
for i=1:SUM+999                        %����N-1��ѭ��
    p(i+1)=u*p(i)*(1-p(i));          %ѭ����������
end
p=p(1001:length(p));

%% ��p���б任��0~255��Χ��������ת����M*N�Ķ�ά����R
p=mod(ceil(p*10^3),256);
R=reshape(p,N,M)';  %ת��M��N��

%% �����緽��
%���ĸ���ֵX0,Y0,Z0,H0
r=(M/t)*(N/t);
X0=0.5008;
Y0=0.5109;
Z0=0.4893;
H0=0.7765;
A=chen_output(X0,Y0,Z0,H0,r);
X=A(:,1);
X=X(1502:length(X));
Y=A(:,2);
Y=Y(1502:length(Y));
Z=A(:,3);
Z=Z(1502:length(Z));
H=A(:,4);
H=H(1502:length(H));

 %X,Y�ֱ����I��R��DNA���뷽ʽ����8�֣�1~8
X=mod(floor(X*10^4),8)+1;
Y=mod(floor(Y*10^4),8)+1;
Z=mod(floor(Z*10^4),3);
Z(Z==0)=3;
Z(Z==1)=0;
Z(Z==3)=1;
H=mod(floor(H*10^4),8)+1;
e=N/t;
%% ͼ����������
YY=imread('/lena.png','png');        %��ȡͼ����Ϣ
YY=double(YY);
Y1=YY(:,:,1);        %R
Y2=YY(:,:,2);        %G
Y3=YY(:,:,3);        %B
MSE_R=zeros(1,21);MSE_G=zeros(1,21);MSE_B=zeros(1,21);
j=0;        %�����±�
for i=0:5:100
    I = imnoise(I, 'gaussian', 0, i^2/255^2);  %�����˹������
    I1=I(:,:,1);     %Rͨ��
    I2=I(:,:,2);     %Gͨ��
    I3=I(:,:,3);     %Bͨ��
    j=j+1;      %�����±�

    %% DNA���루���ܣ�
    for ii=r:-1:2
        Q1_R=DNA_bian(fenkuai(t,I1,ii),H(ii));
        Q1_G=DNA_bian(fenkuai(t,I2,ii),H(ii));
        Q1_B=DNA_bian(fenkuai(t,I3,ii),H(ii));

        Q1_last_R=DNA_bian(fenkuai(t,I1,ii-1),H(ii-1));
        Q1_last_G=DNA_bian(fenkuai(t,I2,ii-1),H(ii-1));
        Q1_last_B=DNA_bian(fenkuai(t,I3,ii-1),H(ii-1));

        Q2_R=DNA_yunsuan(Q1_R,Q1_last_R,Z(ii));        %��ɢǰ
        Q2_G=DNA_yunsuan(Q1_G,Q1_last_G,Z(ii));
        Q2_B=DNA_yunsuan(Q1_B,Q1_last_B,Z(ii));

        Q3=DNA_bian(fenkuai(t,R,ii),Y(ii));

        Q4_R=DNA_yunsuan(Q2_R,Q3,Z(ii));
        Q4_G=DNA_yunsuan(Q2_G,Q3,Z(ii));
        Q4_B=DNA_yunsuan(Q2_B,Q3,Z(ii));

        xx=floor(ii/e)+1;
        yy=mod(ii,e);
        if yy==0
            xx=xx-1;
            yy=e;
        end
        Q_R((xx-1)*t+1:xx*t,(yy-1)*t+1:yy*t)=DNA_jie(Q4_R,X(ii));
        Q_G((xx-1)*t+1:xx*t,(yy-1)*t+1:yy*t)=DNA_jie(Q4_G,X(ii));
        Q_B((xx-1)*t+1:xx*t,(yy-1)*t+1:yy*t)=DNA_jie(Q4_B,X(ii));
    end
    Q5_R=DNA_bian(fenkuai(t,I1,1),H(1));
    Q5_G=DNA_bian(fenkuai(t,I2,1),H(1));
    Q5_B=DNA_bian(fenkuai(t,I3,1),H(1));

    Q6=DNA_bian(fenkuai(t,R,1),Y(1));

    Q7_R=DNA_yunsuan(Q5_R,Q6,Z(1));
    Q7_G=DNA_yunsuan(Q5_G,Q6,Z(1));
    Q7_B=DNA_yunsuan(Q5_B,Q6,Z(1));

    Q_R(1:t,1:t)=DNA_jie(Q7_R,X(1));
    Q_G(1:t,1:t)=DNA_jie(Q7_G,X(1));
    Q_B(1:t,1:t)=DNA_jie(Q7_B,X(1));
    
    Q1=Q_R;
    Q2=Q_G;
    Q3=Q_B;
    
    %ȥ������ʱ������
    if M1~=0
        Q1=Q1(1:M-t+M1,:);
        Q2=Q2(1:M-t+M1,:);
        Q3=Q3(1:M-t+M1,:);
    end
    if N1~=0
        Q1=Q1(:,1:N-t+N1);
        Q2=Q2(:,1:N-t+N1);
        Q3=Q3(:,1:N-t+N1);
    end
    [MM,NN]=size(Q1);     %���»�ý��ܺ��ͼƬ��С
    for m=1:MM
        for n=1:NN
            MSE_R(j)=MSE_R(j)+(Y1(m,n)-Q1(m,n))^2;       %Rͨ��MSE
            MSE_G(j)=MSE_G(j)+(Y2(m,n)-Q2(m,n))^2;       %Gͨ��MSE
            MSE_B(j)=MSE_B(j)+(Y3(m,n)-Q3(m,n))^2;       %Bͨ��MSE
        end
    end
%     RESULT(:,:,1)=uint8(Q_R);
%     RESULT(:,:,2)=uint8(Q_G);
%     RESULT(:,:,3)=uint8(Q_B);
%     figure;imshow(RESULT);title(['��˹��������Ϊ',num2str(i),'ʱ�Ľ���ͼ��']);
end
%��������-MSE
MSE_R=MSE_R./SUM;
MSE_G=MSE_G./SUM;
MSE_B=MSE_B./SUM;
%��ֵ�����-PSNR
PSNR_R=10*log10((255^2)./MSE_R);
PSNR_G=10*log10((255^2)./MSE_G);
PSNR_B=10*log10((255^2)./MSE_B);
%% ��ͼ����������-MSE����ֵ�����-PSNR
X=0:5:100;
figure;plot(X,MSE_R);set(gca,'xtick', X);xlabel('��˹��������');ylabel('�������MSE');title('Rͨ������˹��������-�������MSE����ͼ');
figure;plot(X,MSE_G);set(gca,'xtick', X);xlabel('��˹��������');ylabel('�������MSE');title('Gͨ������˹��������-�������MSE����ͼ');
figure;plot(X,MSE_B);set(gca,'xtick', X);xlabel('��˹��������');ylabel('�������MSE');title('Bͨ������˹��������-�������MSE����ͼ');
figure;plot(X,PSNR_R);set(gca,'xtick', X);xlabel('��˹��������');ylabel('��ֵ�����PSNR��dB��');title('Rͨ������˹��������-��ֵ�����PSNR����ͼ');
figure;plot(X,PSNR_G);set(gca,'xtick', X);xlabel('��˹��������');ylabel('��ֵ�����PSNR��dB��');title('Gͨ������˹��������-��ֵ�����PSNR����ͼ');
figure;plot(X,PSNR_B);set(gca,'xtick', X);xlabel('��˹��������');ylabel('��ֵ�����PSNR��dB��');title('Bͨ������˹��������-��ֵ�����PSNR����ͼ');