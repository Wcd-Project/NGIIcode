本程序主要是保护用户隐私。

观看实验结果，直接运行Main_Test.m文件。


	（1）读取用户数据信息，我们选择我们团队采集的人脸图像数据库（FD-XJ为测试对象），其中
	001_A.bmp 是FD-XJ数据集中的图像。Img是读取数据库中图像。

		Img=imread('001_A.bmp','bmp');   % Original image
        	% Img=imread('lena.png','png');
        	% Img=imread('Yi.jpg','jpg');
	

	（2）加密算法主程序，其中img是加密算法的输出结果，即密文域图像；KEY是算法秘钥；xK是算法参数。

	  	[img,KEY,xK]=main_jiami(Img);    % Encrypted image
 

	（3）解密算法主程序，其中I_yuanTu是解密算法的输出结果。
		[I_yuanTu]=main_jiemi(img,KEY,xK);    % Decrypted image
 


	（4）解密算法主程序，其中NPCR and UACI是算法客观指标。
		% get NPCR and UACI scores
		results = NPCR_and_UACI( Img, img, 3, 255 );

	（5）解密算法主程序，其中MSE,PSNR是算法客观指标。
		% get MSE and PSNR scores
		[MSE,PSNR]=PSNR_MSE(Img) 
 
 


	运行程序之后，既能看到原始的图像，加密图像，以及加密图像的直方图分析，相关性分析，阻塞攻击，解密结果,

以及NPCR、UACI、MSE和PSNR指标。


