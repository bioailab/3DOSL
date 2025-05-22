function x=add_poisson_noise_image(im_name,max_sig,offset,save_im_name,varargin)
disp(version)
max_sig=double(max_sig);
offset=double(offset);

if nargin==5
    imageStack=varargin{1};
else
    ext=fliplr(strtok(fliplr(im_name),'.'));
    if strcmpi(ext,'mat')
        load(im_name);
    elseif strcmpi(ext(1:3),'tif')
        info = imfinfo(im_name);
        imageStack = [];
        numberOfImages = length(info);
        for k = 1:numberOfImages
            currentImage = imread(im_name, k, 'Info', info);
            currentImage=im2double(currentImage);
            imageStack(:,:,k) = currentImage;
        end
%         max_sig=max(imageStack(:));
    end
end
[~,rem]=strtok(fliplr(im_name),'.');
im_name=[save_im_name];
imageStack=double(imageStack);
imageStack=uint16((max_sig-offset)*imageStack/max(imageStack(:))+offset);
imageStack=imnoise(imageStack,'poisson');
save(im_name)
im_name=[im_name];
imwrite(imageStack, save_im_name)
x=1;