import os
import pickle
import numpy as np
import calibration
import scipy.io as sio



base = os.path.relpath('Calibration datasets/Calibration test 19-09-23/data1')


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
x, rms = calib.iterativeCalibraion(eq, Jeq)

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
print('\nResults for a total of {} images:'.format(pts.shape[0]))
print('\u03BC_RP = {} mm'.format(mu_RP))
print('Mean RMS error of equations = {} mm'.format(rms))