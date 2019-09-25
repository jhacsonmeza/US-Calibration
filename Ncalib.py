import os
import numpy as np
import calibration
import scipy.io as sio
import itertools


base = os.path.relpath('Calibration datasets/Calibration test 19-09-23/')
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
c = np.array([161.,204.5]) # Center of image
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
    T_T_W = np.load(os.path.join(path,'target_pose.npy'))
    pts = sio.loadmat(os.path.join(path,'crossP.mat'))['crossP']


    # Set input data and calibrate
    calib.setData(pts, T_T_W)
    eq, Jeq = calib.calibEquations(f, J)
    x, error = calib.iterativeCalibraion(eq, Jeq)
    rms = np.append(rms, error)
    
    
    # Get optimal parameters necessary for quality evaluation
    sx, sy = x[9], x[10]
    T_I_T = calib.T(x[3], x[4], x[5], x[6], x[7], x[8])
    
    # Save optimal paramters
    xhat.append([x[9], x[10], x[3], x[4], x[5], x[6], x[7], x[8]])
    
    # Reconstruct different points of the US image in the target/probe frame
    rc.append(T_I_T @ np.array([sx*c[0],sy*c[1],0,1])) # centre
    rtr.append(T_I_T @ np.array([sx*tr[0],sy*tr[1],0,1])) # top right pixel
    rtl.append(T_I_T @ np.array([sx*tl[0],sy*tl[1],0,1])) # top left pixel
    rbr.append(T_I_T @ np.array([sx*br[0],sy*br[1],0,1])) # bottom right point
    rbl.append(T_I_T @ np.array([sx*bl[0],sy*bl[1],0,1])) # bottom left point


print('\n-> Results for a total of {} calibrations:'.format(len(paths)))


# Report mean RMS error of all equations
print('\nMean RMS error of equations = {} mm'.format(rms.mean()))


# Calculate calibration reproducibility precision at the five image points
rc = np.array(rc)
errc = np.linalg.norm(rc-rc.mean(0),axis=1)

rtr = np.array(rtr)
errtr = np.linalg.norm(rtr-rtr.mean(0),axis=1)

rtl = np.array(rtl)
errtl = np.linalg.norm(rtl-rtl.mean(0),axis=1)

rbr = np.array(rbr)
errbr = np.linalg.norm(rbr-rbr.mean(0),axis=1)

rbl = np.array(rbl)
errbl = np.linalg.norm(rbl-rbl.mean(0),axis=1)

mu_CR1_mean = np.array([errc.mean(), errtr.mean(), errtl.mean(), errbr.mean(),
                        errbl.mean()]).mean()
print('\n\u03BC_CR1 at center = {} mm'.format(errc.mean()))
print('\u03BC_CR1 at bottom right = {} mm'.format(errbr.mean()))
print('\u03BC_CR1 mean = {} mm'.format(mu_CR1_mean))


# Calculate precision with all possible pairs of calibrations in each point
errc2 = np.array([np.linalg.norm(p1-p2) for p1, p2 in 
                  itertools.combinations(rc,2)])
errtr2 = np.array([np.linalg.norm(p1-p2) for p1, p2 in 
                   itertools.combinations(rtr,2)])
errtl2 = np.array([np.linalg.norm(p1-p2) for p1, p2 in 
                   itertools.combinations(rtl,2)])
errbr2 = np.array([np.linalg.norm(p1-p2) for p1, p2 in 
                   itertools.combinations(rbr,2)])
errbl2 = np.array([np.linalg.norm(p1-p2) for p1, p2 in 
                   itertools.combinations(rbl,2)])

mu_CR2_mean = np.array([errc2.mean(), errtr2.mean(), errtl2.mean(), 
                        errbr2.mean(), errbl2.mean()]).mean()
print('\n\u03BC_CR2 at center = {}'.format(errc2.mean()))
print('\u03BC_CR2 at bottom right corner = {}'.format(errbr2.mean()))
print('\u03BC_CR2 mean = {} mm'.format(mu_CR2_mean))



# Estimate final parameters
xhat = np.array(xhat)
xhatm = xhat.mean(0) # Mean of all calibration parameters of each calibration

sxm, sym = xhatm[0], xhatm[1]
T_I_Tm = calib.T(xhatm[2], xhatm[3], xhatm[4], xhatm[5], xhatm[6], xhatm[7])

# Save calibration results
sio.savemat(os.path.join(base,'USparams.mat'), 
            {'sx':sxm,'sy':sym,'T_I_T':T_I_Tm})