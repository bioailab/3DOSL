function x=add_poisson_noise_image(im_names,max_sigs,offsets,varargin)
noisy_stack = zeros(numel(im_names), 128, 128, 'uint16' );
%{
t = Tiff('/mnt/nas1/apu010/data/EMPIAR-10791/data/no_hole.build/1857/5_multi_view_stack_epi_backup/matlab.tif', 'w');
fiji_descr = ['ImageJ=1.52p' newline ...
            'images=1' newline... 
            'channels=1' newline...
            'slices=3' newline...
            'frames=1' newline... 
            'hyperstack=false' newline...
            'mode=grayscale' newline...  
            'loop=false' newline...  
            'min=0.0' newline...      
            'max=65535.0'];  % change this to 256 if you use an 8bit image


tagstruct.ImageLength = 128;
tagstruct.ImageWidth = 128;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 16;
tagstruct.SamplesPerPixel = 1;
% tagstruct.RowsPerStrip = 16;

tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Compression = Tiff.Compression.LZW;
tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
tagstruct.ImageDescription = fiji_descr;
% tagstruct.Software = 'MATLAB';

% tagstruct % display tagstruct
setTag(t,tagstruct);
%}
for i = 1:length(im_names)
    max_sig=max_sigs(i);
    offset=offsets(i);
    im_name = char(im_names(i));
    if nargin==4
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
        end
    end
%    [~,rem]=strtok(fliplr(im_name),'.');
%    im_name=[save_im_name];
    imageStack=double(imageStack);
    imageStack=uint16((max_sig-offset)*imageStack/max(imageStack(:))+offset);
    imageStack=imnoise(imageStack,'poisson');
 %   save(im_name);
 %   im_name=[im_name];
    noisy_stack(i, :,:) = imageStack;
end

%imwrite(noisy_stack, save_im_name, 'tiff', 'ColorSpace', 'cielab');
%write(t,noisy_stack);
    %if i == 1
    %    imwrite(imageStack, save_im_name, 'tiff', 'ColorSpace', 'icclab');
    %else 
    %    imwrite(imageStack, save_im_name, 'tiff', 'ColorSpace', 'icclab', 'WriteMode','append');
    %end
    %disp(i)
    %t.setTag(tagstruct)
    %t.write(im2uint16(imageStack));
    %t.writeDirectory();

%close(t);
x=noisy_stack;