clear, clc

imlist = dir('Calibration test 25-04-19\acquisitionUS\US\*.jpg');

for i = 1:numel(imlist)
    im = imread([imlist(i).folder '\' imlist(i).name]);
    imc = imcrop(im, [293.5 67.5 229 399]); %[381.5 86.5 294 513]
    imwrite(imc, ['Calibration test 25-04-19\acquisitionUS\UScrop\' ...
        imlist(i).name])
end

