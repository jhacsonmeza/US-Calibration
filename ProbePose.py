import os
import cv2
import glob
import target
import pickle
import numpy as np


# Root path
base = os.path.relpath('Calibration datasets/Calibration test 19-09-12/data1')


# Set window name and size
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 640*2, 512)

# Read left and right images of the target
I1 = sorted(glob.glob(os.path.join(base,'L','*bmp')), key=os.path.getctime)
I2 = sorted(glob.glob(os.path.join(base,'R','*bmp')), key=os.path.getctime)

# Load stereo calibration parameters
Params = np.load(os.path.join(os.path.dirname(base),'cam1_cam2.npz'))
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

axes = 40*np.array([[1.,0,0], [0,1.,0], [0,0,1.]]) # axes for drawAxes


T_P_W = []
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
    
    # Rearrange c2 in order to match the points with c1
    c2 = target.match(c1, c2, F)
    
    
    
    # Estimate 3D coordinate of the concentric circles through triangulation
    X = cv2.triangulatePoints(P1, P2, c1, c2)
    X = X[:3]/X[-1] # Convert coordinates from homogeneous to Euclidean
    
    
    # Label the 3D coordinates of the center of each concentric circle as:
    # Xo (origin of target frame), Xx (point in x-axis direction), and
    # Xy (point in y-axis direction).
    Xo, Xx, Xy = target.label(X)
    
    # Target pose estimation
    Rmat, tvec = target.getPose(Xo, Xx, Xy)
    
    # Save pose
    T_P_W.append(np.r_[np.c_[Rmat, tvec], [[0,0,0,1]]])
    
    
    ############################ Visualize results ############################
    # 2D coordinate of origin in image 1
    org1 = P1 @ np.r_[Xo, 1]
    org1 = org1[:2]/org1[-1]
    
    # Draw axes in the first image
    rvec, _ = cv2.Rodrigues(Rmat)
    axs, _ = cv2.projectPoints(axes, rvec, tvec, K1, None)
    img = target.drawAxes(im1.copy(), org1, axs)
    
    cv2.imshow('Detection',np.hstack([img,im2]))
    if cv2.waitKey(500) & 0xFF == 27:
        break
        

cv2.destroyAllWindows()

# Save variables
with open(os.path.join(base,'probe_pose.pkl'), 'wb') as file:
    pickle.dump(T_P_W, file)