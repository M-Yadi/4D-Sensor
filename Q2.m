% Programmed by Mona yadi July, 22, 2019
clear
close all
tic
Image=rgb2gray(imread('Img.jpg')) ; % Loading an Image and Changing colorspace to grayscale; The programm can be easily modified to work for RGB pictures too
N_Image=imnoise(Image,'salt & pepper',0.03); % adding noise to the image
Image=double(Image);
N_Image=double(N_Image);
%%
% % % Noise funtion
% N_Image = zeros(m,n);
% prob=0.01 ;
% thres = 1 - prob;
% for i =1:m
%     for j =1:n
%         rdn =random('unif',0,1);
%         if rdn < prob
%             N_Image(i,j) = 0;
%         elseif rdn > thres
%             N_Image(i,j) = 255;
%         else
%         N_Image(i,j) = Image(i,j);
%         end
%     end
% end

%%
[m, n]=size(N_Image); %extracting the size of the image
figure,
subplot(2,1,1) , imshow(uint8(Image)) ; title('original image');
subplot(2,1,2) ,imshow(uint8(N_Image)) ;title('Noisy image');

%  input2 = padarray(input,[f f],'symmetric');
%% --------------------------------- 

kernel1=1/4*[0 1 0;1 0 1;0 1 0] ; % making the kernel
N4_Image=N_Image;
Con1p = convolve2d(N4_Image,kernel1); %avarage matrix

Con1 =  N4_Image - Con1p; %differentiating
for i =1:m 
        for j=1:n 
            if Con1(i,j)>10 || Con1(i,j)<-10  % If it is too different
                N4_Image(i,j)=Con1p(i,j); %change the value
            end
        end
end

%% -------------------------- 8-Neigbor Filteration

kernel2=1/9*[1 1 1;1 1 1;1 1 1] ; % making the second kernel
N8_Image = zeros(m + 2, n + 2);   % Add zero padding to the input image
N8_Image(2:m+1, 2: n+1) = N_Image; % replacing the zero matrix with the image

Con2p = convolve2d(N8_Image,kernel2); %average matrix
Con3 = N8_Image - Con2p; %differentiating

for i =2:m+1 %starting from 2 because of zero padding
        for j =2:n+1
            if Con3(i,j)>10 || Con3(i,j)<-10  %If it is too different
                % Sorting Brightnesses and their coorditions within the image
                Inf2=[N8_Image(i-1,j-1), i-1, j-1;  N8_Image(i-1,j), i-1, j;  N8_Image(i-1,j+1), i-1, j+1;...
                    N8_Image(i,j-1), i, j-1;  N8_Image(i,j), i, j;  N8_Image(i,j+1),i,j+1; ...
                    N8_Image(i+1,j-1), i+1, j-1;  N8_Image(i+1,j), i+1, j;  N8_Image(i+1,j+1), i+1, j+1];
                Inf2=sortrows(Inf2); % sorting the values with respect to the brightnesses
                avToSub=sum(Inf2(3:7))/5; %calculating the avarage of 5 remaining brightness
                N8_Image(Inf2(1,2),Inf2(1,3))=avToSub; %replacing the avarage with the corrupted pixel
                N8_Image(Inf2(2,2),Inf2(2,3))=avToSub; %replacing the avarage with the corrupted pixel
                N8_Image(Inf2(8,2),Inf2(8,3))=avToSub; %replacing the avarage with the corrupted pixel
                N8_Image(Inf2(9,2),Inf2(9,3))=avToSub;%replacing the avarage with the corrupted pixel
            end
        end
end
N8_Image=N8_Image(2:m+1, 2: n+1); %removing zero padding
toc
% Preparation for depicting images (Stacking)
figure,
subplot(2,1,1) , imshow(uint8(N4_Image)) ; title('4-nei');
subplot(2,1,2) ,imshow(uint8(N8_Image)) ;title('8-nei');


 function output=convolve2d(image, kernel) % For convolving the kernel and the image
[m, n]=size(image); %extracting the size of the image
 output = zeros(m,n); % Convolution output
image_padded = zeros(m + 2, n + 2);   % Add zero padding to the input image
image_padded(2:m+1, 2: n+1) = image; % replacing the zero matrix with the image
for x=1:n      % Loop over every pixel of the image
    for y=1:m
        output(y,x)=sum(sum(kernel.*image_padded(y:y+2,x:x+2)));
    end
end
end

 
 




