clear, clc

imlist = dir('Calibration test 25-04-19\acquisitionUS\UScrop\*.jpg');

crossP = zeros(numel(imlist), 2);
for i = 1:numel(imlist)
    im = imread([imlist(i).folder '\' imlist(i).name]);
%     figure, imagesc(im), colormap gray, axis off image
    figure, imshow(im)
    [x,y] = ginput(1);
    close
    crossP(i,:) = [x y];
end

