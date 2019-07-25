clear, clc

p = 80*[0 1 0 0;0 0 1 0;0 0 0 1];

figure, hold on, axis equal, xlabel('x'), ylabel('y'), zlabel('z'), grid on

% Cam1
R = eye(3);
t = zeros(3,1);

ax = R*p+t;
plot3([ax(1,1) ax(1,2)],[ax(2,1) ax(2,2)],[ax(3,1) ax(3,2)],'r')
plot3([ax(1,1) ax(1,3)],[ax(2,1) ax(2,3)],[ax(3,1) ax(3,3)],'g')
plot3([ax(1,1) ax(1,4)],[ax(2,1) ax(2,4)],[ax(3,1) ax(3,4)],'b')
text(ax(1,1), ax(2,1), ax(3,1), '\{W\}', 'FontSize',10)




% Cam 2
R = [0.89595941, -0.00101544, -0.44413478;
    -0.00585225,  0.99988358, -0.01409188;
    0.44409739,  0.01522494,  0.89584916];
t = [297.58578984,   8.94221492,  69.16131674]';

ax = R*p+t;
plot3([ax(1,1) ax(1,2)],[ax(2,1) ax(2,2)],[ax(3,1) ax(3,2)],'r')
plot3([ax(1,1) ax(1,3)],[ax(2,1) ax(2,3)],[ax(3,1) ax(3,3)],'g')
plot3([ax(1,1) ax(1,4)],[ax(2,1) ax(2,4)],[ax(3,1) ax(3,4)],'b')
text(ax(1,1), ax(2,1), ax(3,1), 'cam2', 'FontSize',10)




% T_P_W[0] <---> US Probe system
R = [-0.70550377, -0.70889311,  0.07315216;
    -0.37833402,  0.28518949, -0.87993176;
    0.59927272, -0.64508721, -0.46940065];
t = [-213.69800943,  -39.1173398 , 1215.92704843]';

ax = R*p+t;
plot3([ax(1,1) ax(1,2)],[ax(2,1) ax(2,2)],[ax(3,1) ax(3,2)],'r')
plot3([ax(1,1) ax(1,3)],[ax(2,1) ax(2,3)],[ax(3,1) ax(3,3)],'g')
plot3([ax(1,1) ax(1,4)],[ax(2,1) ax(2,4)],[ax(3,1) ax(3,4)],'b')
text(ax(1,1), ax(2,1), ax(3,1), '\{P\}', 'FontSize',10)




% T_I_W = T_P_W[0] @ T_I_P <---> US image system
R = [0.73880779,  0.66075613,  0.15225192;
    0.40845439, -0.25429886, -0.87592619;
    -0.53661198,  0.70574342, -0.45779479];
t = [-177.39826582,  -51.77130556, 1315.42710246]';

ax = R*p+t;
plot3([ax(1,1) ax(1,2)],[ax(2,1) ax(2,2)],[ax(3,1) ax(3,2)],'r')
plot3([ax(1,1) ax(1,3)],[ax(2,1) ax(2,3)],[ax(3,1) ax(3,3)],'g')
plot3([ax(1,1) ax(1,4)],[ax(2,1) ax(2,4)],[ax(3,1) ax(3,4)],'b')
text(ax(1,1), ax(2,1), ax(3,1), '\{I\}', 'FontSize',10)





% np.linalg.inv(T_W_C) <---> Cross-wire system
R = eye(3);
t = [-130.90632354,  -55.78801192, 1338.47638298]';

ax = R*p+t;
plot3([ax(1,1) ax(1,2)],[ax(2,1) ax(2,2)],[ax(3,1) ax(3,2)],'r')
plot3([ax(1,1) ax(1,3)],[ax(2,1) ax(2,3)],[ax(3,1) ax(3,3)],'g')
plot3([ax(1,1) ax(1,4)],[ax(2,1) ax(2,4)],[ax(3,1) ax(3,4)],'b')
text(ax(1,1), ax(2,1), ax(3,1), '\{C\}', 'FontSize', 10)
pcshow(ax(:,1)', 'VerticalAxis', 'y', 'VerticalAxisDir', 'down')

hold off
% xlim([-300, 400]);
% ylim([-80, 10]);
% zlim([0, 1360]);
camorbit(0, -30);