clc;
clear all;
close all;
N=21;
n=0:1:N-1;


%Unit Step Sequence

x=ones(1,N);

subplot(3,2,1);stem(n,x);
xlabel('n');ylabel('x(n)');
title('Unit Step Sequence');


%Exponential Sequence

x2=0.8 .^(n);

subplot(3,2,2);stem(n,x2);
xlabel('n');ylabel('x(n)');
title('Exponential Sequence');


% Ramp Sequence

x = 10;

t=0:x;

subplot(3,2,3);stem(t,t);
xlabel('c'); ylabel('Amplitude');
title(' Ramp Sequence');


%Sinusoidal sequence

t=0:0.01:4;
y=sin(2*pi*t);

subplot(3,2,4); stem(t,y);
xlabel('e'); ylabel('Amplitude');
title('Sinusoidal Sequence');


% Cosine Sequence

t=0:0.01:4;
y=cos(2*pi*t);

subplot(3,2,5); stem(t,y);
xlabel('f'); ylabel('Amplitude');
title('Cosine Sequence');
