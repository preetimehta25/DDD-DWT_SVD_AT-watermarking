%DDD-DWT_SVD.m

clc 
close all 
clear all 

 %% option 1: watermarking an image (select the folder to pick images)

change directory
folder_name = uigetdir(pwd, 'Select Directory Where the .m Files Reside');
if ( folder_name ~= 0 )
    if ( strcmp(pwd, folder_name) == 0 )
        cd(folder_name);
    end
else
    return;
end
%% option 2: watermarking an image (select the path of folder to pick images)

image_folder = 'E:\iclDataset\SingleCaptureImages\D40'; %  Enter name of folder from which you want to upload pictures with full path
filenames = dir(fullfile(image_folder, '*.jpg'));  % read all images with specified extention, its jpg in our case
total_images = numel(filenames);    % count total number of photos present in that folder
%total_images =1; %% mention the number of images to be watermarked (optional)

display('Loading Images...')
tic
folder = 'E:\watermarking_SVD_DTCWT\watermarking_MATLABcodes_arnoldmapping\watermark_images';%Outputfolder name for storing watermarked images

%% Embedding process code begins

% Watermak Image
watermark=imread('wmark1','bmp'); 
watermark=imresize(watermark,[256 256]);  
watermark=im2bw(watermark);  
figure,imshow(watermark),title('watermark image')
[rows,columns,c]=size(watermark);

% Frame by Frame Embedding

display('Begin Embedding...')
tic


% cover_object=double(imread(file_name)); 
for frame_no = 1:total_images

% full_name= fullfile(image_folder, filenames(frame_no).name);         % it will specify images names with full path and extension
% cover_object = imread(full_name);    
cover_object = imread('peppers.png'); %%testing for one image
cover_object = rgb2gray(cover_object);
[mm,nn,c]=size(cover_object);
cover_object = double( cover_object  ) ;
cover_object = imresize(cover_object, [512 512], 'bilinear');
%     R_in = cover_object(:,:,1); %% for RGB images, pick blue frame to
%                                     watermark
%     G_in = cover_object(:,:,2);
%     B_in = cover_object(:,:,3);

B_in = cover_object ;
figure,imshow((cover_object),[]);

%psuedo matrix generation using a key

key = 3;
rng(key,'combRecursive')
%rand('seed', key);
% produce binary sequence matrix to perform XOR with the arnold watermark
% image
binary_mask = randi([0 1], size(watermark,1),size(watermark,2));
%figure,imshow(binary_mask)
N = 256;

oldScrambledImage = watermark; %original watermark image
%arnold scrambing of watermark image of size 256x256
iteration_num = 1; %never equal to 384 for 512 size image and 
while iteration_num <=10
for i = 1 : rows % y
		for k = 1 : columns % x
            
			c = mod((2 * k) + i, N) + 1; % x coordinate
			r = mod(k + i, N) + 1; % y coordinate
			% Move the pixel.  Note indexes are (row, column) = (y, x) NOT (x, y)!
			currentScrambledImage(i, k, :) = oldScrambledImage(r, c, :);
            
		end
end
oldScrambledImage = currentScrambledImage;
         %   figure,imshow(currentScrambledImage)
          iteration_num = iteration_num+1;
end

    figure,imshow(currentScrambledImage)
    
signature = double( bitxor(uint8(currentScrambledImage), uint8(binary_mask)) ); % signature length=512

W=signature;
figure;
subplot(1,2,1); 
imshow(watermark,[]); 
title('Watermark'); 
subplot(1,2,2); 
imshow(W,[]); 
title('Encrypted Watermark'); 

%%%%%%%%%%%%  SVD-DDD-DWT

DF = dtfilters('dtf1');
wt1 = dddtree2('cplxdt',B_in,4,DF{1},DF{2});

a15 = wt1.cfs{1}(:,:,1,1,1);
a45 = wt1.cfs{1}(:,:,2,1,1);
a75 = wt1.cfs{1}(:,:,3,1,1);
b15 = wt1.cfs{1}(:,:,1,2,1);
b45 = wt1.cfs{1}(:,:,2,2,1);
b75= wt1.cfs{1}(:,:,3,2,1);

% figure;
% subplot(321),imshow(a15);
% subplot(322),imshow(a45);
% subplot(323),imshow(a75);
% subplot(324),imshow(b15);
% subplot(325),imshow(b45);
% subplot(326),imshow(b75);

% SVD on a75 subband of the Cover Image
%D=dct2(a75);
[U,S,V]=svd(a75);  
af=1;  
[lm,ln]=size(a75); 
%[Uw,Sw,Vw]=svd(W);%encrypted watermark
[Uw,Sw,Vw]=svd(W);%direct watermark
% % SVD on Encrypted Watermark Image
% WW=zeros(lm,ln);
% for i=1:rows 
%     for j=1:columns
%             WW(i,j)=W(i,j); 
%     end 
% end 
Temp=af*Sw;
% [U1,S1,V1]=svd(Temp); 

% Inverse SVD Of Watermarked Image
CW=U*Temp*V'; 

% Inverse DWT

wt1.cfs{1}(:,:,3,1,1) = CW ;
% I1=idwt2(CW,HL,LH,HH,'haar'); 
I1 = idddtree2(wt1);
%RGB_out = uint8(cat(3,R_in, G_in, I1));
CWI = I1;
%CWI = RGB_out;
 figure;
 imshow(CWI,[])
% imshow(RGB_out,[]); 
 title(' Watermarked Image'); 


display(['Embedded ' num2str(frame_no) ' images successfully...']);
% corr_val(frame_no) = psnr((RGB_out),uint8(cover_object));
% ssimval(frame_no) = ssim(uint8(cover_object),RGB_out);

outputFileName = fullfile(folder, ['stegnoImage' num2str(frame_no) '.png']);
%imwrite(RGB_out, outputFileName);
end
toc
 

%% %%%%%%%%%%%%%%%%%% Attacks for testing %%%%%%%%%%%%%%%%%%%%%%%

attackd=1; 
while attackd~=0 

disp('Attacks on Watermarked Image : '); 
disp('0--·Watermarked Image without Attack');  
disp('1--Cropping');    
disp('2--Adjust Image Intensity: 0.4 to 1');    
disp('3--Adjust Image Intensity: 0 to 0.95 ');    
disp('4--Adjust Image Intensity: 0.2 to 0.6'); 
disp('5--Adjust Image Intensity: 0.2 to 0.8'); 
disp('6--Speckle : Multiplicative noise'); 
disp('7--Gaussian Noise'); 
disp('8--Rotate 45 Degree');  
disp('9--Histogram Equalization');  
disp('10--Salt & Pepper Noise');
disp('11--[3 3] gaussian filter'); 
disp('12--Resize'); 
disp('13--JPEG Image '); 
disp('14--Motion Blurred Image');

attackd=input('Input your Choice (1-14):'); 

 f=CWI; 
switch attackd 
  case 0, 
    attackf=f; 
    att='Watermarked Image without Attack'; 
    
case 1, 
%%1. Cropping

f(200:320,200:320)=512; 
attackf=f; 
att='Cropping'; 
break; 

  case 2, 
%%2. Adjust image intensity values:0.4-1
attackf=imadjust(f,[],[0.4,1]); 
att='Adjust Image Intensity: 0.4 to 1'; 
break; 

  case 3, 
%%3. Adjust image intensity values:0-0.95
attackf=imadjust(f,[],[0,0.95]); 
att='Adjust Image Intensity: 0 to 0.95'; 
break; 
  case 4, 
      
%%4.Adjust image intensity values:0.2-0.6
attackf=imadjust(f,[0.2,0.6],[]); 
att='Adjust Image Intensity: 0.2 to 0.6'; 
break; 

  case 5, 
%%5.Adjust image intensity values:0.2-0.8
attackf=imadjust(f,[],[0.2,0.8]); 
att='Adjust Image Intensity: 0.2 to 0.8'; 
break; 

  case 6, 
%%6. Speckle : Multiplicative noise
attackf=imnoise(f,'speckle',0.01); 
att='Speckle : Multiplicative noise'; 
break; 

case 7, 
%%7. Gaussian Noise 
attackf=imnoise(f,'gaussian',0,0.01); 
att='Gaussian Noise '; 
break; 
 
case 8 
%%8.Rotate 45 Degree
attackf=imrotate(f,45,'bilinear','crop'); 
att='Rotate 45 Degree'; 
break; 

case 9 
%%9.Histogram Equalization
attackf=histeq(f); 
att='Histogram Equalization'; 
break; 
 

case 10 
attackf=imnoise(f,'salt & pepper',0.01);
att='0.01 salt & pepper noise'; 
break;

case 11
H=fspecial('gaussian',[3,3],1); 
attackf=imfilter(f,H);     
att='through filter [10,10] '; 
break;

case 12
 attackf1=imresize(f,2); 
 attackf=imresize(attackf1,1/2); 
 att='Resize'; 
 break;
 
case 13
q=input('Quality factor:'); 
imwrite(f,'watermarked_image.bmp','jpg','quality',q); 
attackf=imread('watermarked_image.bmp'); 
att='JPEG Image';  
break;

case 14
    
H = fspecial('motion',20,45); 
attackf = imfilter(f,H,'replicate'); 
att='Motion Blurred Image of watermaked Image'; 
break;
end;

attackd=1; 
end 
 
figure(4); 
imshow(attackf);2

title(att);

%% %%%%%%% EXTRACTION OF WATERMARK %%%%%%%%%%%%%%

display('Begin Extracting...')
tic

af = 1;
% image_folder_o = 'E:\iclDataset\SingleCaptureImages\D40'; %  Enter name of folder from which you want to upload pictures with full path
% filenames_o = dir(fullfile(image_folder_o, '*.jpg'));  % read all images with specified extention, its jpg in our case
%  
image_folder = 'E:\watermarking_SVD_DTCWT\watermarking_MATLABcodes_arnoldmapping\watermark_images'; %  Enter name of folder from which you want to upload pictures with full path
 filenames = dir(fullfile(image_folder, '*.png'));  % read all images with specified extention, its jpg in our case
  total_images = numel(filenames);    % count total number of photos present in that folder
%q=input('Quality factor:'); 
%total_images = 2;
 for frame_no = 1:total_images
     
%      full_name_o= fullfile(image_folder_o, filenames_o(frame_no).name);         % it will specify images names with full path and extension
% cover_object = imread(full_name_o);    
% [mm,nn,c]=size(cover_object);
% cover_object = double( cover_object  ) ;
% cover_object = imresize(cover_object, [512 512], 'bilinear');

 full_name= fullfile(image_folder, filenames(frame_no).name);         % it will specify images names with full path and extension
orig_image = (imread(full_name));                 % Read images  
[mm,nn,rr] = size(orig_image);

 
 %figure,imshow((orig_image));
%    attackf=histeq(orig_image); 
af_pic = imrotate(orig_image,1);
  orig_image=af_pic;
 figure,imshow(orig_image,[]);
 orig_image = imresize(orig_image, [512 512], 'bilinear');
%  corr_val(frame_no) = psnr(uint8(orig_image),uint8(cover_object));
% ssimval(frame_no) = ssim(uint8(cover_object),orig_image);

 orig_image = double(orig_image);
  
    R_in = orig_image(:,:,1);
    G_in = orig_image(:,:,2);
    B_in = orig_image(:,:,3);
        
 DF = dtfilters('dtf1');
wt1 = dddtree2('cplxdt',B_in,1,DF{1},DF{2});
wa75 = wt1.cfs{1}(:,:,3,1,1);
%[LL1 HL1 LH1 HH1]=dwt2(CWW,'haar'); 

[U2,S2,V2]=svd(wa75); 
  
S_old=(S2)/af;  

WNN = Uw*S_old*Vw';
%figure,imshow(WNN), title('extracted encrypted signature')


key = 3;
rng(key,'combRecursive')
%rand('seed', key);
% produce binary sequence matrix to perform XOR with the arnold watermark
% image
binary_mask = randi([0 1], size(watermark,1),size(watermark,2));
signature_extracted = double( bitxor(uint8(WNN), uint8(binary_mask)) ); % signature length=512
%figure,imshow(signature_extracted)

fontSize = 20;
[rows, columns] = size(signature_extracted);
% Inverse Arnold Map

N =256;
% iteration = 11; %never equal to 384 for 512 size image and 
oldScrambledImage = signature_extracted;

for iteration = 11: 192
	% Scramble the image based on the old image.
	for row = 1 : rows % y
		for col = 1 : columns % x
			c = mod((2 * col) + row, N) + 1; % x coordinate
			r = mod(col + row, N) + 1; % y coordinate
			% Move the pixel.  Note indexes are (row, column) = (y, x) NOT (x, y)!
			currentScrambledImage(row, col, :) = oldScrambledImage(r, c, :);
		end
	end
	
	% Display the current image.
% 	caption = sprintf('Arnolds Cat Map, Iteration #%d', iteration);
% 	fprintf('%s\n', caption);
%     figure(21)
% 	imshow(currentScrambledImage);
% 	axis on;
% 	title(caption, 'FontSize', fontSize);
% 	drawnow;
	
	% Insert a delay if desired.
% 	pause(0.1);
	
	% Save the image, if desired.
% 	filename = sprintf('Arnold Cat Iteration %d.png', iteration);
% % 	imwrite(currentScrambledImage, filename);
% 	fprintf('Saved image file %s to disk.\n', filename);
	
	if corr2(uint8(currentScrambledImage), uint8(watermark)) >=90
		caption = sprintf('Back to Original after %d Iterations.', iteration);
		fprintf('%s\n', caption);
		title(caption, 'FontSize', fontSize);
		break;
	end
	
	% Make the current image the prior/old one so we'll operate on that the next iteration.
	oldScrambledImage = currentScrambledImage;
	% Update the iteration counter.
	iteration = iteration+1;
end
figure
imshow(currentScrambledImage,[]); 

NC(frame_no)=corr2(watermark,currentScrambledImage); 


fprintf('Normalized Correlation between Embedded and Extracted Watermark is:%5.4f\n',NC); 

 end
 toc
