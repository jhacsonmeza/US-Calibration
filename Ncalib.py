import os
import pickle
import numpy as np
import calibration
import scipy.io as sio
import itertools


base = os.path.relpath('Calibration datasets/Calibration test 19-09-12/')
paths = [os.path.join(base, x) for x in os.listdir(base) if 'data' in x]


# Create calibration object
calib = calibration.Calibration()

# Model construction
f, J = calib.model()


'''# US image points to evaluate. Probe depth: 7 cm.
c = np.array([115.,200.]) # Center of image
tl = np.array([0.,0.]) # Top left point
tr = np.array([230.,0.]) # Top right point
bl = np.array([0.,400.]) # Bottom left point
br = np.array([230.,400.]) # Bottom right point'''

# US image points to evaluate. Probe depth: 5 cm.
c = np.array([151.,204.5]) # Center of image
tl = np.array([0.,0.]) # Top left point
tr = np.array([322.,0.]) # Top right point
bl = np.array([0.,409.]) # Bottom left point
br = np.array([322.,409.]) # Bottom right point

rc = []
rtr = []
rtl = []
rbr = []
rbl = []

rms = np.array([])
xhat = []
for i, path in enumerate(paths):
    print('calibration {}/{}'.format(i+1,len(paths)))
    
    # load known variables
    with open(os.path.join(path,'probe_pose.pkl'),'rb') as file:
        T_P_W = pickle.load(file)

    pts = sio.loadmat(os.path.join(path,'crossP.mat'))['crossP']


    # Set input data and calibrate
    calib.setData(pts, T_P_W)
    eq, Jeq = calib.calibEquations(f, J)
    x, error = calib.iterativeCalibraion(eq, Jeq)
    rms = np.append(rms, error)
    
    
    # Get optimal parameters needed for quality evaluation    
    sx, sy = x[9], x[10]
    T_I_P = calib.T(x[3], x[4], x[5], x[6], x[7], x[8])
    
    # Save optimal paramters
    xhat.append([x[9], x[10], x[3], x[4], x[5], x[6], x[7], x[8]])
    
    
    # Compute centre of US image in the probe frame
    rc.append(T_I_P @ np.array([sx*c[0],sy*c[1],0,1]))
    
    # Compute top right point of US image in the probe frame
    rtr.append(T_I_P @ np.array([sx*tr[0],sy*tr[1],0,1]))
    
    # Compute top left point of US image in the probe frame
    rtl.append(T_I_P @ np.array([sx*tl[0],sy*tl[1],0,1]))
    
    # Compute bottom right point of US image in the probe frame
    rbr.append(T_I_P @ np.array([sx*br[0],sy*br[1],0,1]))
    
    # Compute bottom left point of US image in the probe frame
    rbl.append(T_I_P @ np.array([sx*bl[0],sy*bl[1],0,1]))


# Report mean RMS error of all equations
print('\nMean RMS error = {} mm with {} calibrations'.format(
        rms.mean(), len(paths)))


# Calculate calibration reproducibility precision at center of image
rc = np.array(rc)
errc = np.linalg.norm(rc-rc.mean(0),axis=1)
mu_CR = errc.mean()
print('\n\u03BC_CR at center = {} mm with {} calibrations'.format(
        mu_CR, len(paths)))

# Calculate calibration reproducibility precision at the four corners of image
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


# Calculate precision with all possible pairs of calibrations at bottom right 
# corner
pre = []
for p1, p2 in itertools.combinations(rbr,2):
    pre.append(np.linalg.norm(p1-p2))

print('\nPrecision at bottom right corner = {}'.format(sum(pre)/len(pre)))


# Estimate final parameters
xhat = np.array(xhat)
xhatm = xhat.mean(0) # Mean of all calibration parameters of each calibration

sxm, sym = xhatm[0], xhatm[1]
T_I_Pm = calib.T(xhatm[2], xhatm[3], xhatm[4], xhatm[5], xhatm[6], xhatm[7])

# Save calibration results
sio.savemat(os.path.join(base,'USparams.mat'), 
            {'sx':sxm,'sy':sym,'T_I_P':T_I_Pm})