import os
import cv2
import glob
import target
import pickle
import numpy as np
import sympy as sym
import scipy.io as sio

base = 'Calibration test 25-04-19/'


##############################################################################
# Create calibration model
##############################################################################

# Variables for rotation matrices and general transformation matrix T_i_j
ax, ay, az, tx, ty, tz = sym.symbols('ax ay az tx ty tz')

Rx = sym.Matrix([[1,0,0],
                [0,sym.cos(ax),-sym.sin(ax)],
                [0,sym.sin(ax),sym.cos(ax)]])

Ry = sym.Matrix([[sym.cos(ay),0,sym.sin(ay)],
                 [0,1,0],
                 [-sym.sin(ay),0,sym.cos(ay)]])

Rz = sym.Matrix([[sym.cos(az),-sym.sin(az),0],
                 [sym.sin(az),sym.cos(az),0],
                 [0,0,1]])

T = (Rz*Ry*Rx).row_join(sym.Matrix([tx,ty,tz])).col_join(
        sym.Matrix([[0,0,0,1]]))


# Variables for known transformation T_s_r
c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34 = \
sym.symbols('c11 c12 c13 c14 c21 c22 c23 c24 c31 c32 c33 c34')

T_s_r = sym.Matrix([[c11,c12,c13,c14],[c21,c22,c23,c24],[c31,c32,c33,c34],
                 [0,0,0,1]])


# Symbolic variables for image coordinates of point phantom
u, v = sym.symbols('u v')

# Symbolic variables for unknowns:
# x10 -> sx, x11 -> sy
# from T_I_S: x4 -> tx, x5 -> ty, x6 -> tz, x7 -> az, x8 -> ay, x9 -> ax
# from T_R_H: x1 -> tx, x2 -> ty, x3 -> tz
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = \
sym.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11')

# Model: T_R_H * T_S_R* T_I_S * [sx*u, sy*v, 0, 1]
f = T.subs({tx:x1,ty:x2,tz:x3,az:0,ay:0,ax:0})*T_s_r*T.subs({tx:x4,ty:x5,
          tz:x6,az:x7,ay:x8,ax:x9})*sym.Matrix([x10*u,x11*v,0,1])

f = sym.Matrix([sym.sqrt(f[0,0]**2+f[1,0]**2+f[2,0]**2)])
#f = f[:3,0]



crossP = sio.loadmat(base+'crossP.mat')['crossP']







cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 640*2, 512)

I1 = sorted(glob.glob(base+'acquisitionUS/L/*.jpg'), key=os.path.getmtime)
I2 = sorted(glob.glob(base+'acquisitionUS/R/*.jpg'), key=os.path.getmtime)

# Load stereo calibration parameters
Params = sio.loadmat(base+'Params.mat')
K1 = Params['K1']
K2 = Params['K2']
R = Params['R']
t = Params['t']
distCoeffs1 = Params['distCoeffs1'][0]
distCoeffs2 = Params['distCoeffs2'][0]

# Create projection matrices of camera 1 and camera 2
P1 = K1 @ np.c_[np.eye(3), np.zeros(3)]
P2 = K2 @ np.c_[R, t]

axes = 40*np.array([[1.,0,0], [0,1.,0], [0,0,1.]]) # axes for drawAxes


eq = sym.Matrix()
Jeq = sym.Matrix()
T_S_R = []
pts =[]
fail = False
for i, (im1n, im2n) in enumerate(zip(I1,I2)):
    im1 = cv2.imread(im1n)
    im2 = cv2.imread(im2n)
    
    # Target detection
    im1, c1, ret1 = target.detection(im1, True)
    im2, c2, ret2 = target.detection(im2, True)
    
    if not (ret1 and ret2):
        print('\nCircles in image {} and {} couldn\'t be detected'.format(
                im1n.split('\\')[-1], im2n.split('\\')[-1]))
        continue
    

    # Target labeling
    im1, org1, i1, j1 = target.label(im1,c1)
    im2, org2, i2, j2 = target.label(im2,c2)
    
    # Matrix of 2D target points
    p1 = np.array([org1,i1,j1]).reshape(-1,1,2)
    p2 = np.array([org2,i2,j2]).reshape(-1,1,2)
    
    # Undistort estimated points
    p1 = cv2.undistortPoints(p1, K1, distCoeffs1, None, None, K1)
    p2 = cv2.undistortPoints(p2, K2, distCoeffs2, None, None, K2)
    p1 = p1.reshape(-1,2).T
    p2 = p2.reshape(-1,2).T
    
    # Target pose estimation
    Rm, tvec = target.targetPose(P1, P2, p1, p2)
    
    T_S_R.append(np.r_[np.c_[Rm, tvec], np.array([[0,0,0,1]])])
    pts.append([crossP[i,0], crossP[i,1]])
    
    eq = eq.col_join(f.subs({u:crossP[i,0],v:crossP[i,1],c11:Rm[0,0],
                             c12:Rm[0,1],c13:Rm[0,2],c14:tvec[0,0],
                             c21:Rm[1,0],c22:Rm[1,1],c23:Rm[1,2],
                             c24:tvec[1,0],c31:Rm[2,0],c32:Rm[2,1],
                             c33:Rm[2,2],c34:tvec[2,0]}))
    
    
    # Draw axes in the first image
    rvec, _ = cv2.Rodrigues(Rm)
    axs, _ = cv2.projectPoints(axes, rvec, tvec, K1, None)
    img = target.drawAxes(im1.copy(), org1, axs)
    
    cv2.imshow('Detection',np.hstack([img,im2]))
    if cv2.waitKey(10) & 0xFF == 27:
        break
        

cv2.destroyAllWindows()
pts = np.array(pts)


with open(base+'model.pkl', 'wb') as file:
    pickle.dump(eq, file)
    pickle.dump(pts, file)
    pickle.dump(T_S_R, file)
#X = cv2.triangulatePoints(P1,P2,p1.reshape(-1,2).T,p2.reshape(-1,2).T)
#X = X[:3]/X[-1]
#np.linalg.norm(X[:,1]-X[:,0])
#np.linalg.norm(X[:,2]-X[:,0])
#
#X = cv2.triangulatePoints(P1,P2,p1u,p2u)
#X = X[:3]/X[-1]
#np.linalg.norm(X[:,1]-X[:,0])
#np.linalg.norm(X[:,2]-X[:,0])