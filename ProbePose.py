import os
import cv2
import yaml
import glob
import target
import numpy as np
import scipy.io as sio


# Load config file
with open('config.yaml','r') as file:
    config = yaml.safe_load(file)


# Root path
base = os.path.relpath(config['root_path'])
paths = [os.path.join(base, x) for x in os.listdir(base) if 'data' in x]


# Set window name and size
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 1792, 717)

# Load stereo calibration parameters
params_file = config['params_file']
if params_file.split('.')[1] == 'npz':
    Params = np.load(os.path.join(base,params_file))
elif params_file.split('.')[1] == 'mat':
    Params = sio.loadmat(os.path.join(base,params_file))

K1 = Params['K1']
K2 = Params['K2']
R = Params['R']
t = Params['t']
F = Params['F']
dist1 = Params['dist1']
dist2 = Params['dist2']

# Create projection matrices of camera 1 and camera 2
P1 = K1 @ np.c_[np.eye(3), np.zeros(3)]
P2 = K2 @ np.c_[R, t]


for path in paths:
    # Read left and right images of the target
    I1 = target.natsort(glob.glob(os.path.join(path,config['left_folder'],'*')))
    I2 = target.natsort(glob.glob(os.path.join(path,config['right_folder'],'*')))

    T_T_W = []
    errs = []
    for im1n, im2n in zip(I1,I2):
        im1 = cv2.imread(im1n)
        im2 = cv2.imread(im2n)
        
        # Target detection
        ret1, im1, c1 = target.detect(im1, global_th=False, th_im=True)
        ret2, im2, c2 = target.detect(im2, global_th=False, th_im=True)
        
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
        errs.append([abs(np.linalg.norm(Xo-Xx)-25),
                    abs(np.linalg.norm(Xo-Xy)-40)])
        
        # Target pose estimation relative to the left camera/world frame
        R_T_W, t_T_W = target.getPose(Xo, Xx, Xy)
        
        # Save pose
        T_T_W.append(np.r_[np.c_[R_T_W, t_T_W], [[0,0,0,1]]])
        
        
        ############################ Visualize results ############################
        target.drawAxes(im1, K1, dist1, R_T_W, t_T_W)
        target.drawEpilines(im2, c1, 1, F)
        # target.drawCub(im1, K1, dist1, R_T_W, t_T_W)
        # target.drawCenters(im1, im2, K1, K2, R, t, dist1, dist2, Xo, Xx, Xy)
        
        cv2.imshow('Detection',np.hstack([im1,im2]))
        if cv2.waitKey(500) & 0xFF == 27:
            break
    
    print('Mean absolute error of distances in x and y: {} mm, from folder {}'.format(
        np.array(errs).mean(0), os.path.basename(path)))

    # Save variables
    np.save(os.path.join(path,'target_pose.npy'),np.dstack(T_T_W).transpose(2,0,1))

cv2.destroyAllWindows()