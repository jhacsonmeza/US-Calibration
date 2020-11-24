import os
import numpy as np
import calibration


# Load config file
with open('config.yaml','r') as file:
    config = yaml.safe_load(file)


base = os.path.relpath(config['root_path'])
paths = [os.path.join(base, x) for x in os.listdir(base) if 'data' in x]


# Create calibration object
calib = calibration.Calibration()

# Model construction
f, J = calib.model()

# US scan size
w, h = config['w'], config['h']

# US image points to evaluate.
c = np.array([w/2,h/2]) # Center of image
tl = np.array([0.,0.]) # Top left point
tr = np.array([w,0.]) # Top right point
bl = np.array([0.,h]) # Bottom left point
br = np.array([w,h]) # Bottom right point

rc = []
rtr = []
rtl = []
rbr = []
rbl = []

rms = np.array([])
xhat = []
for i, path in enumerate(paths):
    print('Calibration {}/{}'.format(i+1,len(paths)))
    
    # load known variables
    T_T_W = np.load(os.path.join(path,'target_pose.npy'))
    pts = np.load(os.path.join(path,'cross_point.npy'))


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


rc = np.array(rc)
rtr = np.array(rtr)
rtl = np.array(rtl)
rbr = np.array(rbr)
rbl = np.array(rbl)

# Save report of calibration assessment
calib.writeReport(os.path.join(base,'report.txt'), rms, rc, rtr, rtl, rbr, rbl)



# Estimate final parameters
xhat = np.array(xhat)
xhatm = xhat.mean(0) # Mean of all calibration parameters of each dataset

sxm, sym = xhatm[0], xhatm[1]
T_I_Tm = calib.T(xhatm[2], xhatm[3], xhatm[4], xhatm[5], xhatm[6], xhatm[7])

# Save calibration results
np.savez(os.path.join(base,'USparams.npz'), sx=sxm, sy=sym, T_I_T=T_I_Tm)