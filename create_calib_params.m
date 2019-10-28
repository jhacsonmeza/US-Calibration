clear, clc
root = 'Datasets/19-06-08/';

load([root,'stereoParams.mat'])

K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
K2 = stereoParams.CameraParameters2.IntrinsicMatrix';
R = stereoParams.RotationOfCamera2';
t = stereoParams.TranslationOfCamera2';
F = stereoParams.FundamentalMatrix;

dist1 = [stereoParams.CameraParameters1.RadialDistortion(1:2) ...
    stereoParams.CameraParameters1.TangentialDistortion ...
    stereoParams.CameraParameters1.RadialDistortion(3)];

dist2 = [stereoParams.CameraParameters2.RadialDistortion(1:2) ...
    stereoParams.CameraParameters2.TangentialDistortion ...
    stereoParams.CameraParameters2.RadialDistortion(3)];

save([root,'Params.mat'],'K1','K2','R','t','F','dist1','dist2')