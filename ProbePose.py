import os
import cv2
import glob
import target
import pickle
import numpy as np
import scipy.io as sio


# Root path
base = 'Calibration test 08-05-19/'


# Set window name and size
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 640*2, 512)

# Read left and right images of the target
I1 = sorted(glob.glob(base+'acquisitionUS/L/*.bmp'), key=os.path.getmtime)
I2 = sorted(glob.glob(base+'acquisitionUS/R/*.bmp'), key=os.path.getmtime)

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


T_P_W = []
for im1n, im2n in zip(I1,I2):
    im1 = cv2.imread(im1n)
    im2 = cv2.imread(im2n)
    
    # Target detection
    ret1, im1, c1 = target.detect(im1)
    ret2, im2, c2 = target.detect(im2)
    
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
    Rmat, tvec = target.getPose(P1, P2, p1, p2)
    
    # Save pose and cross-wire coordinates
    T_P_W.append(np.r_[np.c_[Rmat, tvec], np.array([[0,0,0,1]])])
    
    # Draw axes in the first image
    rvec, _ = cv2.Rodrigues(Rmat)
    axs, _ = cv2.projectPoints(axes, rvec, tvec, K1, None)
    img = target.drawAxes(im1.copy(), org1, axs)
    
    cv2.imshow('Detection',np.hstack([img,im2]))
    if cv2.waitKey(500) & 0xFF == 27:
        break
        

cv2.destroyAllWindows()

# Save variables
with open(base+'probe_pose.pkl', 'wb') as file:
    pickle.dump(T_P_W, file)