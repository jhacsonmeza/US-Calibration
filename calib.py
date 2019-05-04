import pickle
import numpy as np
import sympy as sym
from scipy.optimize import root

base = 'Calibration test 11-04-19/'

def T(tx, ty, tz, az, ay, ax):
    Rx = np.array([[1,0,0],
                   [0,np.cos(ax),-np.sin(ax)],
                   [0,np.sin(ax),np.cos(ax)]])
    
    Ry = np.array([[np.cos(ay),0,np.sin(ay)],
                    [0,1,0],
                    [-np.sin(ay),0,np.cos(ay)]])
    
    Rz = np.array([[np.cos(az),-np.sin(az),0],
                    [np.sin(az),np.cos(az),0],
                    [0,0,1]])
    
    return np.r_[np.c_[Rz@Ry@Rx, np.array([tx,ty,tz])], np.array([[0,0,0,1]])]


# load model file
with open(base+'model.pkl','rb') as file:
    f = pickle.load(file)
    Jf = pickle.load(file)

# load known variables
with open(base+'known_variables.pkl','rb') as file:
    pts = pickle.load(file)
    T_P_W = pickle.load(file)





##############################################################################
##############################################################################
# Compute a least squares initial estimation of the US calibration:
# min||Ax-b|| where A = [ui*R_P_Wi, vi*R_P_Wi, R_P_Wi, -I] is a 3n x 12 matrix
# (n>=4), x = [sx*R_I_P[:,0], sy*R_I_P[:,1], t_I_P, t_W_C] is a 12 x 1 vector 
# and b = [-t_P_W] is a 3n x 1 vector
##############################################################################
##############################################################################

A = []
b = []
for ptsi, T_P_Wi in zip(pts, T_P_W):
    
    R_P_W, t_P_W = T_P_Wi[:3,:3], T_P_Wi[:3,-1]

    A.append(np.c_[ptsi[0]*R_P_W, ptsi[1]*R_P_W, R_P_W, -np.eye(3)])
    b.append(-t_P_W)


A = np.vstack(A)
b = np.array(b).flatten()
x = np.linalg.pinv(A) @ b

# Get scale factors
sx = np.linalg.norm(x[:3])
sy = np.linalg.norm(x[3:6])

# Get rotation angles from R_I_P
r1 = x[:3]/sx
r2 = x[3:6]/sy
r3 = np.cross(r1, r2)
R_I_P = np.c_[r1, r2, r3] # is not necessarily a rotation matrix
                          # the orthonormality constraints were not enforced

# get the closest (Frobenius norm) rotation matrix via SVD
U, S, Vh = np.linalg.svd(R_I_P)
R_I_P = U @ Vh.T

# extract the Euler angles
ay = np.arctan2(-R_I_P[2,0], np.sqrt(R_I_P[0,0]**2 + R_I_P[1,0]**2))

smallAngle = 0.5*np.pi/180
if abs(ay-np.pi/2) > smallAngle and abs(ay+np.pi/2) > smallAngle:
    az = np.arctan2(R_I_P[1,0]/np.cos(ay), R_I_P[0,0]/np.cos(ay))
    ax = np.arctan2(R_I_P[2,1]/np.cos(ay), R_I_P[2,2]/np.cos(ay))
else:
    az = 0
    ax = np.arctan2(R_I_P[0,1], R_I_P[1,1])
 
# Final result:
x0 = [x[9], x[10], x[11], x[6], x[7], x[8], az, ay, ax, sx, sy]
print('\nInitial parameters = \n{}'.format(x0))






##############################################################################
##############################################################################
# Compute an iterative least squares solution of the US calibration with the 
# Levenberg-Marquardt algorithm, in order to refine the above initial
# estimation
##############################################################################
##############################################################################


# Symbolic unknowns of the model
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = \
sym.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11')

# Convert model from symbolic to lambda function
f = sym.lambdify([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], f)
Jf = sym.lambdify([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], Jf)


# Objective function that returns function value and jacobian matrix evaluated
# in x, which is passed to scipy root function
def objfunc(x):
    return f(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],
               x[10]).T.tolist()[0], \
             Jf(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10])


# Solve problem using Levenberg-Marquardt algorithm
sol = root(objfunc, x0, jac=True, method='lm',
           options={'ftol':1e-16,'xtol':1e-16,'gtol':1e-16,'maxiter':5000, 
                    'eps':1e-19})
print('\nRefined parameters = \n{}'.format(sol.x.tolist()))

sx, sy = sol.x[9], sol.x[10]
T_I_P = T(sol.x[3], sol.x[4], sol.x[5], sol.x[6], sol.x[7], sol.x[8])
T_W_C = T(sol.x[0], sol.x[1], sol.x[2], 0, 0, 0)



err = []
for i in range(len(pts)):
    phat = T_W_C @ T_P_W[i] @ T_I_P @ np.array([sx*pts[i,0],sy*pts[i,1],0,1])
    
    err.append(np.linalg.norm(phat[:3]))

print('\nPrecision = {} mm'.format(sum(err)/len(err)))