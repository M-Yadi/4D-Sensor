% Programmed by Mona yadi July, 22, 2019
clc; 
clear; close all;
Image_size=500;
Grating_pitch=20;
Grating1D=zeros(500,500);
Grating2D=zeros(500,500);


%----------------------------------Cos waves / Soft edges
t= 0:Image_size-1 ; %time
a= cos(2*pi*1/Grating_pitch.*t)*255/2; %1D
A=a+255/2;
for i =1:numel(t)
   Grating1D(i,:)=A;
end

Grating2D=Grating1D.*Grating1D.'; % To obtain 2D pattern 
Grating2D=Grating2D./max(max(Grating2D))*255 ;% Normalization and rescaling to 0-255

figure,
subplot(1,2,1) , imshow(uint8(Grating1D)) ; title('Gratings 1D soft edges');
subplot(1,2,2) ,imshow(uint8(Grating2D)) ;title('Gratings 2D soft edges');
%------------------Hard edges---------- As question %1 does not describe that if it needs hard or soft edges / I have considered hard edges too
Grating_pitch=20;
Image_size=500;

a=ones(Image_size,Image_size);
y=round(Image_size/Grating_pitch);
for h =1:y
    a(:,h*20:h*20+10)=0;
end
a=a(1:Image_size,1:Image_size);
c=a.';
g=a.*c; 

a=a*255; %1D Hard edges
g=g*255; %2D Hard edges


figure,
subplot(1,2,1) , imshow(uint8(a)) ; title('Gratings 1D hard edges');
subplot(1,2,2) ,imshow(uint8(g)) ;title('Gratings 2D hard edges');


