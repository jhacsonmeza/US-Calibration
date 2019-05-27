import pickle
import numpy as np
import calibration
import scipy.io as sio



base = 'Calibration test 21-05-19 part1/'


# load known variables
with open(base+'probe_pose.pkl','rb') as file:
    T_P_W = pickle.load(file)

pts = sio.loadmat(base+'crossP.mat')['crossP']


# Calibration object
calib = calibration.Calibration(pts, T_P_W)



# Model construction
f, J = calib.model()
eq, Jeq = calib.calibEquations(f, J)
x = calib.iterativeCalibraion(eq, Jeq)

sx, sy = x[9], x[10]
T_I_P = calib.T(x[3], x[4], x[5], x[6], x[7], x[8])
T_W_C = calib.T(x[0], x[1], x[2], 0, 0, 0)

err = []
for i in range(len(pts)):
    phat = T_W_C @ T_P_W[i] @ T_I_P @ np.array([sx*pts[i,0],sy*pts[i,1],0,1])
    
    err.append(np.linalg.norm(phat[:3]))

print('\nPrecision = {} mm'.format(sum(err)/len(err)))