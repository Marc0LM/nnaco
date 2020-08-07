%m=csvread('IrisTestaco-bp 218 58867101.csv');
%m=csvread('CancerTestaco-bp 220 14075060.csv');
%m=csvread('MPGTestaco-bp 218 52169937.csv');
m=csvread('Last.csv');
s=size(m);
it=5;
numExp=4;
t=downsample(m(1,:),it);
b=zeros(numExp,s(2)/it);
ab=zeros(numExp,s(2)/it);
hab=zeros(numExp,s(2)/it);
for i=1:numExp
    ab(i,:)=downsample(mean(m(0*numExp+1+(i-1)*4+1:0*numExp+4+(i-1)*4+1,:)),it); 
    b(i,:)=downsample(mean(m(4*numExp+1+(i-1)*4+1:4*numExp+4+(i-1)*4+1,:)),it); 
    hab(i,:)=downsample(mean(m(8*numExp+1+(i-1)*4+1:8*numExp+4+(i-1)*4+1,:)),it); 
end
figure(1);
plot(t(1,:),ab(4,:),':k',t(1,:),b(4,:),'-k',t(1,:),hab(4,:),'--k');
figure(2);
bo=[m(0*numExp+1+(i-1)*4+1:0*numExp+4+(i-1)*4+1,s(2)),m(4*numExp+1+(i-1)*4+1:4*numExp+4+(i-1)*4+1,s(2)),m(8*numExp+1+(i-1)*4+1:8*numExp+4+(i-1)*4+1,s(2))];
boxplot(bo);