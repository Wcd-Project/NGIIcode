function fv=fenkuai(t,I,num)
[~,N]=size(I);
N=N/t;
x=floor(num/N)+1;      %��
y=mod(num,N);           %��
if y==0
    x=x-1;
    y=N;
end
fv=I(t*(x-1)+1:t*x,t*(y-1)+1:t*y);


