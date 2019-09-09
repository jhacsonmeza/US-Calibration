import os
import pickle
import numpy as np
import calibration
import scipy.io as sio



base = os.path.relpath('Calibration datasets/Calibration test 19-05-08/data1')


# load known variables
with open(os.path.join(base,'probe_pose.pkl'),'rb') as file:
    T_P_W = pickle.load(file)

pts = sio.loadmat(os.path.join(base,'crossP.mat'))['crossP']


# Create calibration object
calib = calibration.Calibration(pts, T_P_W)

# Model construction
f, J = calib.model()

# Equations and Jacobian matrix construction
eq, Jeq = calib.calibEquations(f, J)

# Calibrate with iterative Levenberg-Marquardt algorithm
x = calib.iterativeCalibraion(eq, Jeq)[0]

# Get results
sx, sy = x[9], x[10]
T_I_P = calib.T(x[3], x[4], x[5], x[6], x[7], x[8])
T_W_C = calib.T(x[0], x[1], x[2], 0, 0, 0)

# Estimate reconstruction precision (RP) named in literature as μ_RP:
rec = []
for i in range(len(pts)):
    phat = T_P_W[i] @ T_I_P @ np.array([sx*pts[i,0],sy*pts[i,1],0,1])
    
    rec.append(phat)

rec = np.array(rec)
err = np.linalg.norm(rec-rec.mean(0),axis=1)
mu_RP = err.mean()
print('\n\u03BC_RP = {} mm, with {} images'.format(mu_RP, pts.shape[0]))


# Filter out points with high reconstruction error
ind = err < 0.5

pts2 = pts[ind]
T_P_W2 = []
for i in range(len(err)):
    if ind[i]:
        T_P_W2.append(T_P_W[i])
        
# Replace data in the object
calib.setData(pts2, T_P_W2)

# Equations and Jacobian matrix construction
eq, Jeq = calib.calibEquations(f, J)

# Calibrate with iterative Levenberg-Marquardt algorithm
x = calib.iterativeCalibraion(eq, Jeq)[0]

# Get results
sx, sy = x[9], x[10]
T_I_P = calib.T(x[3], x[4], x[5], x[6], x[7], x[8])
T_W_C = calib.T(x[0], x[1], x[2], 0, 0, 0)

# Estimate reconstruction precision (RP) named in literature as μ_RP:
rec2 = []
for i in range(len(pts2)):
    phat = T_P_W2[i] @ T_I_P @ np.array([sx*pts2[i,0],sy*pts2[i,1],0,1])
    
    rec2.append(phat)

rec2 = np.array(rec2)
err2 = np.linalg.norm(rec2-rec2.mean(0),axis=1)
mu_RP2 = err2.mean()
print('\n\u03BC_RP = {} mm, with {} images'.format(mu_RP2, pts2.shape[0]))