% K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
% K2 = stereoParams.CameraParameters2.IntrinsicMatrix';
% R = stereoParams.RotationOfCamera2';
% t = stereoParams.TranslationOfCamera2';
% save Params.mat K1 K2 R t

clear, clc

load pts2D
load stereoParams
for i = 1:size(pim1, 1)
    mtch1 = [pim1(i,1:2); pim1(i,3:4); pim1(i,5:6)];
    mtch2 = [pim2(i,1:2); pim2(i,3:4); pim2(i,5:6)];
    X = triangulate(mtch1,mtch2,stereoParams)';
end