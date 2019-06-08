import glob
import os
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

paths = ['Calibration test 19-06-01 part2/',
         'Calibration test 19-06-01 part3/',
         'Calibration test 19-06-01 part4/',
         'Calibration test 19-06-01 part5/',
         'Calibration test 19-06-01 part6/',
         'Calibration test 19-06-01 part7/',
         'Calibration test 19-06-01 part8/',
         'Calibration test 19-06-01 part9/',
         'Calibration test 19-06-01 part10/',
         'Calibration test 19-06-01 part12/',
         'Calibration test 19-06-01 part13/',
         'Calibration test 19-06-01 part14/',
         'Calibration test 19-06-01 part17/',
         'Calibration test 19-06-01 part18/',
         'Calibration test 19-06-01 part19/',
         'Calibration test 19-06-01 part20/']

paths = ['F:/Calibration test 19-06-05 part1/',
         'F:/Calibration test 19-06-05 part2/']




folders = sorted(glob.glob('Calibration test 19-06-08\\*'), 
                 key=os.path.getmtime)

#paths = ['Calibration test 19-06-08/part1/',
#         'Calibration test 19-06-08/part2/',
#         'Calibration test 19-06-08/part3/',
#         'Calibration test 19-06-08/part4/']


paths = ['Calibration test 19-06-08/part1/',
         'Calibration test 19-06-08/part2/',
         'Calibration test 19-06-08/part3/',
         'Calibration test 19-06-08/part4/',
#         'Calibration test 19-06-08/mala - part2/',
         'Calibration test 19-06-08/mala - part3/',
#         'Calibration test 19-06-08/mala - part4/',
         'Calibration test 19-06-08/mala - part4 2/',
         'Calibration test 19-06-08/mala - part4 3/',
         'Calibration test 19-06-08/Calibration test 19-06-07/']




# Calibration object
calib = calibration.Calibration()
f, J = calib.model()





c = np.array([115.,200.])
tr = np.array([0.,0.])
tl = np.array([230.,0.])
br = np.array([0.,400.])
bl = np.array([230.,400.])

rc = []
rtr = []
rtl = []
rbr = []
rbl = []

#rms = []
rms = np.array([])
for i, base in enumerate(paths):
    print('Process {}/{}'.format(i+1,len(paths)))
    
    # load known variables
    with open(base+'probe_pose.pkl','rb') as file:
        T_P_W = pickle.load(file)

    pts = sio.loadmat(base+'crossP.mat')['crossP']


    # Set input data and calibrate
    calib.setData(pts, T_P_W)
    eq, Jeq = calib.calibEquations(f, J)
    x, error = calib.iterativeCalibraion(eq, Jeq)
#    print(x)
    rms = np.append(rms, error)
    
    
    sx, sy = x[9], x[10]
    T_I_P = calib.T(x[3], x[4], x[5], x[6], x[7], x[8])
    
    
    # Compute centre in the probe frame
    phat = T_I_P @ np.array([sx*c[0],sy*c[1],0,1])
    rc.append(phat)
    
    # Compute top right in the probe frame
    phat = T_I_P @ np.array([sx*tr[0],sy*tr[1],0,1])
    rtr.append(phat)
    
    # Compute top left in the probe frame
    phat = T_I_P @ np.array([sx*tl[0],sy*tl[1],0,1])
    rtl.append(phat)
    
    # Compute bottom right in the probe frame
    phat = T_I_P @ np.array([sx*br[0],sy*br[1],0,1])
    rbr.append(phat)
    
    # Compute bottom left in the probe frame
    phat = T_I_P @ np.array([sx*bl[0],sy*bl[1],0,1])
    rbl.append(phat)

rc = np.array(rc)
errc = np.linalg.norm(rc-rc.mean(0),axis=1)
mu_CR = errc.mean()
print('\n\u03BC_CR at center = {} mm with {} calibrations'.format(
        mu_CR, len(paths)))


rtr = np.array(rtr)
errtr = np.linalg.norm(rtr-rtr.mean(0),axis=1)

rtl = np.array(rtl)
errtl = np.linalg.norm(rtl-rtl.mean(0),axis=1)

rbr = np.array(rbr)
errbr = np.linalg.norm(rbr-rbr.mean(0),axis=1)

rbl = np.array(rbl)
errbl = np.linalg.norm(rbl-rbl.mean(0),axis=1)

mu_CR_mean = np.array([errc.mean(), errtr.mean(), errtl.mean(), errbr.mean(), 
                       errbl.mean()]).mean()

print('\n\u03BC_CR mean = {} mm with {} calibrations'.format(
        mu_CR_mean, len(paths)))


print('\nMean RMS error = {} mm with {} calibrations'.format(
        rms.mean(), len(paths)))


pre = []
for p1, p2 in itertools.combinations(rc,2):
    pre.append(np.linalg.norm(p1-p2))

print('\nPrecision = {}'.format(sum(pre)/len(pre)))