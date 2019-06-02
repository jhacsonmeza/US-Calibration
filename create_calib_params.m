% clear, clc
% root = 'Calibration test 30-04-19\';
% 
% load([root,'stereoParams.mat'])
% 
% K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
% K2 = stereoParams.CameraParameters2.IntrinsicMatrix';
% R = stereoParams.RotationOfCamera2';
% t = stereoParams.TranslationOfCamera2';
% distCoeffs1 = [stereoParams.CameraParameters1.RadialDistortion ...
%     stereoParams.CameraParameters1.TangentialDistortion];
% distCoeffs2 = [stereoParams.CameraParameters2.RadialDistortion ...
%     stereoParams.CameraParameters2.TangentialDistortion];
% 
% save([root,'Params.mat'],'K1','K2','R','t','distCoeffs1','distCoeffs2')

clear, clc
root = 'Calibration test 08-05-19\';

load([root,'stereoParams3.mat'])

K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
K2 = stereoParams.CameraParameters2.IntrinsicMatrix';
R = stereoParams.RotationOfCamera2';
t = stereoParams.TranslationOfCamera2';

distCoeffs1 = [stereoParams.CameraParameters1.RadialDistortion(1:2) ...
    stereoParams.CameraParameters1.TangentialDistortion ...
    stereoParams.CameraParameters1.RadialDistortion(3)];

distCoeffs2 = [stereoParams.CameraParameters2.RadialDistortion(1:2) ...
    stereoParams.CameraParameters2.TangentialDistortion ...
    stereoParams.CameraParameters2.RadialDistortion(3)];

save([root,'Params3.mat'],'K1','K2','R','t','distCoeffs1','distCoeffs2')