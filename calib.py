import pickle
import numpy as np
import sympy as sym
from scipy.optimize import root

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
with open('model.pkl','rb') as file:
    f = pickle.load(file)
    Jf = pickle.load(file)
    pts = pickle.load(file)
    T_S_R = pickle.load(file)




x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = \
sym.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11')

f = sym.lambdify([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], f)
Jf = sym.lambdify([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], Jf)


def objfunc(x):
    return f(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],
               x[10]).T.tolist()[0], \
    Jf(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10])


# Solve problem using Levenberg-Marquardt algorithm
#sol = root(objfunc, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], jac=True, method='lm')
sol = root(objfunc, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], jac=True, method='lm',
           options={'xtol':1e-20,'ftol':1e-20,'maxiter':50000})
print('\nCalculated parameters = \n{}'.format(sol.x))

sx, sy = sol.x[:2]
T_I_S = T(sol.x[2], sol.x[3], sol.x[4], sol.x[5], sol.x[6], sol.x[7])
T_R_H = T(sol.x[8], sol.x[9], sol.x[10], 0, 0, 0)



err = []
for i in range(len(pts)):
    phat = T_R_H @ T_S_R[i] @ T_I_S @ np.array([sx*pts[i,0],sy*pts[i,1],0,1])
    
    err.append(np.linalg.norm(phat[:3]-np.array([0,0,0])))



#f.subs({u:crossP[i,1],v:crossP[i,0],c11:Rm[0,0],c12:Rm[0,1],c13:Rm[0,2],
#        c14:tvec[0,0],c21:Rm[1,0],c22:Rm[1,1],c23:Rm[1,2],c24:tvec[1,0],
#        c31:Rm[2,0],c32:Rm[2,1],c33:Rm[2,2],c34:tvec[2,0],x1:sol.x[0],
#        x2:sol.x[1],x3:sol.x[2],x4:sol.x[3],x5:sol.x[4],x6:sol.x[5],
#        x7:sol.x[6],x8:sol.x[7],x9:sol.x[8],x10:sol.x[9],x11:sol.x[10]})