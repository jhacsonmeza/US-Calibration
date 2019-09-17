import os
import cv2
import glob
import target
import pickle
import numpy as np
import scipy.io as sio


# Root path
base = os.path.relpath('Calibration datasets/Calibration test 19-09-12/data2')


# Set window name and size
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 640*2, 512)

# Read left and right images of the target
I1 = target.natsort(glob.glob(os.path.join(base,'L','*bmp')))
I2 = target.natsort(glob.glob(os.path.join(base,'R','*bmp')))

# Load stereo calibration parameters
Params = np.load(os.path.join(os.path.dirname(base),'cam1_cam2.npz'))
#Params = sio.loadmat(os.path.join(os.path.dirname(base),'Params.mat'))
K1 = Params['K1']
K2 = Params['K2']
R = Params['R']
t = Params['t']
F = Params['F']
dist1 = Params['dist1'][0]
dist2 = Params['dist2'][0]

# Create projection matrices of camera 1 and camera 2
P1 = K1 @ np.c_[np.eye(3), np.zeros(3)]
P2 = K2 @ np.c_[R, t]


T_P_W = []
errs = []
for im1n, im2n in zip(I1,I2):
    im1 = cv2.imread(im1n)
    im2 = cv2.imread(im2n)
    
    # Target detection
    ret1, im1, c1 = target.detect(im1)
    ret2, im2, c2 = target.detect(im2)
    
    if not (ret1 and ret2):
        print('\nCircles in image {} or {} couldn\'t be detected'.format(
                im1n.split('\\')[-1], im2n.split('\\')[-1]))
        continue
    
    
    # Undistort 2D center coordinates in each image
    c1 = cv2.undistortPoints(c1, K1, dist1, None, None, K1)
    c2 = cv2.undistortPoints(c2, K2, dist2, None, None, K2)
    
    
    # Estimate unlabel 3D coordinates of centers
    X = target.centers3D(P1, P2, c1, c2)
    
    # Label the 3D coordinates of the center of each concentric circle as:
    # Xo (origin of target frame), Xx (point in x-axis direction), and
    # Xy (point in y-axis direction).
    Xo, Xx, Xy = target.label(X)
    errs.append([abs(np.linalg.norm(Xo-Xx)-25)/25,
                 abs(np.linalg.norm(Xo-Xy)-40)/40])
    
    # Target pose estimation
    Rmat, tvec = target.getPose(Xo, Xx, Xy)
    
    # Save pose
    T_P_W.append(np.r_[np.c_[Rmat, tvec], [[0,0,0,1]]])
    
    
    ############################ Visualize results ############################
#    target.drawAxes(im1, K1, dist1, Rmat, tvec)
#    target.drawCub(im1, K1, dist1, Rmat, tvec)
    target.drawCenters(im1, im2, K1, K2, R, t, dist1, dist2, Xo, Xx, Xy)
    
    cv2.imshow('Detection',np.hstack([im1,im2]))
    if cv2.waitKey(500) & 0xFF == 27:
        break
        

cv2.destroyAllWindows()

# Save variables
with open(os.path.join(base,'probe_pose.pkl'), 'wb') as file:
    pickle.dump(T_P_W, file)