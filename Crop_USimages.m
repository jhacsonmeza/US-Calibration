clear, clc

imlist = dir('acquisitionUS\US\*.jpg');

for i = 1:numel(imlist)
    im = imread([imlist(i).folder '\' imlist(i).name]);
    imc = imcrop(im, [381.5 86.5 294 513]);
    imwrite(imc, ['acquisitionUS\UScrop\'  imlist(i).name])
end

