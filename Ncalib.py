import pickle
import numpy as np
import calibration
import scipy.io as sio
import itertools



paths = ['Calibration test 19-06-01 part1/',
         'Calibration test 19-06-01 part2/',
         'Calibration test 19-06-01 part3/',
         'Calibration test 19-06-01 part4/',
         'Calibration test 19-06-01 part5/',
         'Calibration test 19-06-01 part6/',
         'Calibration test 19-06-01 part7/',
         'Calibration test 19-06-01 part8/',
         'Calibration test 19-06-01 part9/',
         'Calibration test 19-06-01 part10/',
         'Calibration test 19-06-01 part11/',
         'Calibration test 19-06-01 part12/',
         'Calibration test 19-06-01 part13/',
         'Calibration test 19-06-01 part14/',
         'Calibration test 19-06-01 part15/',
         'Calibration test 19-06-01 part16/',
         'Calibration test 19-06-01 part17/',
         'Calibration test 19-06-01 part18/',
         'Calibration test 19-06-01 part19/',
         'Calibration test 19-06-01 part20/']

paths = ['Calibration test 19-05-21 part1/',
         'Calibration test 19-05-21 part2/',
         'Calibration test 19-05-21 part3/',
         'Calibration test 19-05-21 part4/',
         'Calibration test 19-05-21 part5/',
         'Calibration test 19-05-08/']


# Calibration object
calib = calibration.Calibration()
f, J = calib.model()




c = np.array([115.,200.])
#c = np.array([230.,400.])
r = []
for base in paths:
    
    # load known variables
    with open(base+'probe_pose.pkl','rb') as file:
        T_P_W = pickle.load(file)

    pts = sio.loadmat(base+'crossP.mat')['crossP']


    # Set input data and calibrate
    calib.setData(pts, T_P_W)
    eq, Jeq = calib.calibEquations(f, J)
    x = calib.iterativeCalibraion(eq, Jeq)

    sx, sy = x[9], x[10]
    T_I_P = calib.T(x[3], x[4], x[5], x[6], x[7], x[8])
    
    
    # Compute p in the probe frame
    phat = T_I_P @ np.array([sx*c[0],sy*c[1],0,1])
    
    r.append(phat)

r = np.array(r)
err = np.linalg.norm(r-r.mean(0),axis=1)
mu_CR = err.mean()
print('\n\u03BC_CR = {} mm, with {} calibrations'.format(mu_CR, len(paths)))


pre = []
for p1, p2 in itertools.combinations(r,2):
    pre.append(np.linalg.norm(p1-p2))