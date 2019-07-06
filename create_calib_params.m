clear, clc
root = 'Calibration test 19-06-08/part1/';

load([root,'stereoParams.mat'])

K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
K2 = stereoParams.CameraParameters2.IntrinsicMatrix';
R = stereoParams.RotationOfCamera2';
t = stereoParams.TranslationOfCamera2';
F = stereoParams.FundamentalMatrix;

distCoeffs1 = [stereoParams.CameraParameters1.RadialDistortion(1:2) ...
    stereoParams.CameraParameters1.TangentialDistortion ...
    stereoParams.CameraParameters1.RadialDistortion(3)];

distCoeffs2 = [stereoParams.CameraParameters2.RadialDistortion(1:2) ...
    stereoParams.CameraParameters2.TangentialDistortion ...
    stereoParams.CameraParameters2.RadialDistortion(3)];

save([root,'Params.mat'],'K1','K2','R','t','F','distCoeffs1','distCoeffs2')