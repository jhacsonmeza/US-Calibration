clear, clc

syms ax ay az tx ty tz

Rx = [1,0,0;0,cos(ax),-sin(ax);0,sin(ax),cos(ax)];
Ry = [cos(ay),0,sin(ay);0,1,0;-sin(ay),0,cos(ay)];
Rz = [cos(az),-sin(az),0;sin(az),cos(az),0;0,0,1];

% General transformation matrix
T(tx,ty,tz,az,ay,ax) = [Rz*Ry*Rx [tx;ty;tz];0 0 0 1];

syms u v
syms x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11
syms c11 c12 c13 c14 c21 c22 c23 c24 c31 c32 c33 c34

Tc = [c11 c12 c13 c14;c21 c22 c23 c24;c31 c32 c33 c34;0 0 0 1];
f = T(x9,x10,x11,0,0,0)*Tc*T(x3,x4,x5,x6,x7,x8)*[x1*u;x2*v;0;1];
f = f(1:3);

% Jacobian matrix
Jf = jacobian(f, [x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11]);