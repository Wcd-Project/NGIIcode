��������Ҫ�Ǳ����û���˽��

�ۿ�ʵ������ֱ������Main_Test.m�ļ���


	��1����ȡ�û�������Ϣ������ѡ�������ŶӲɼ�������ͼ�����ݿ⣨FD-XJΪ���Զ��󣩣�����
	001_A.bmp ��FD-XJ���ݼ��е�ͼ��Img�Ƕ�ȡ���ݿ���ͼ��

		Img=imread('001_A.bmp','bmp');   % Original image
        	% Img=imread('lena.png','png');
        	% Img=imread('Yi.jpg','jpg');
	

	��2�������㷨����������img�Ǽ����㷨������������������ͼ��KEY���㷨��Կ��xK���㷨������

	  	[img,KEY,xK]=main_jiami(Img);    % Encrypted image
 

	��3�������㷨����������I_yuanTu�ǽ����㷨����������
		[I_yuanTu]=main_jiemi(img,KEY,xK);    % Decrypted image
 


	��4�������㷨����������NPCR and UACI���㷨�͹�ָ�ꡣ
		% get NPCR and UACI scores
		results = NPCR_and_UACI( Img, img, 3, 255 );

	��5�������㷨����������MSE,PSNR���㷨�͹�ָ�ꡣ
		% get MSE and PSNR scores
		[MSE,PSNR]=PSNR_MSE(Img) 
 
 


	���г���֮�󣬼��ܿ���ԭʼ��ͼ�񣬼���ͼ���Լ�����ͼ���ֱ��ͼ����������Է������������������ܽ��,

�Լ�NPCR��UACI��MSE��PSNRָ�ꡣ


