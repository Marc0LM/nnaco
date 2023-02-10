%m=csvread('IrisTestaco-bp 218 58867101.csv');
%m=csvread('CancerTestaco-bp 220 14075060.csv');
%m=csvread('MPGTestaco-bp 218 52169937.csv');

function plott(path,isC,iExp,numExp)
m=csvread(path);
s=size(m);
it=10;
t=downsample(m(1,:),it);
b=zeros(numExp,s(2)/it);
ab=zeros(numExp,s(2)/it);
hab=zeros(numExp,s(2)/it);

for i=1:numExp
    ab(i,:)=downsample(mean(m(0*numExp+1+(i-1)*4+1:0*numExp+4+(i-1)*4+1,:)),it); 
    b(i,:)=downsample(mean(m(4*numExp+1+(i-1)*4+1:4*numExp+4+(i-1)*4+1,:)),it); 
    hab(i,:)=downsample(mean(m(8*numExp+1+(i-1)*4+1:8*numExp+4+(i-1)*4+1,:)),it); 
end
y=[ab(iExp,:);b(iExp,:);hab(iExp,:)];

if(isC)
    figure(1); 
    plotf(t,y);
else
    figure(2); 
    plotfreg(t,y);
end
end
%figure(2);
%bo=[ab(1:numExp,s(2)/it),b(1:numExp,s(2)/it),hab(1:numExp,s(2)/it)];
%boxplot(bo);