import numpy as np
import sympy as sym
from scipy.optimize import root
from scipy.spatial.transform import Rotation


class Calibration:
    def __init__(self, pts, T_P_W):

        self.setData(pts, T_P_W)
        
        # Variables for known transformation T_P_W
        self.c11, self.c12, self.c13, self.c14 = sym.symbols('c11 c12 c13 c14')
        self.c21, self.c22, self.c23, self.c24 = sym.symbols('c21 c22 c23 c24')
        self.c31, self.c32, self.c33, self.c34 = sym.symbols('c31 c32 c33 c34')
        
        # Symbolic variables for image coordinates of point phantom
        self.u, self.v = sym.symbols('u v')
        
        # Symbolic variables for unknowns:
        # from T_W_C: tx -> x1, ty -> x2, tz -> x3
        # from T_I_P: tx -> x4, ty -> x5, tz -> x6, az -> x7, ay -> x8, ax -> x9
        # sx -> x10, sy -> x11
        self.x1, self.x2, self.x3 = sym.symbols('x1 x2 x3') 
        self.x4, self.x5, self.x6 = sym.symbols('x4 x5 x6') 
        self.x7, self.x8, self.x9 = sym.symbols('x7 x8 x9') 
        self.x10, self.x11 = sym.symbols('x10 x11')
    
    
    def setData(self, pts, T_P_W):
        self.pts = pts
        self.T_P_W = T_P_W
    
    
    def T(self, tx, ty, tz, az, ay, ax):
        R = Rotation.from_euler('xyz',[ax,ay,az]).as_dcm()
        
        return np.r_[np.c_[R, [tx,ty,tz]], np.array([[0,0,0,1]])]
        
    
    
    def model(self):
        # rotation matrices and general transformation matrix T_i_j
        
        # Variables for rotation matrices and general transformation 
        # matrix T_i_j
        ax, ay, az, tx, ty, tz = sym.symbols('ax ay az tx ty tz')
        
        Rx = sym.Matrix([[1,0,0],
                         [0,sym.cos(ax),-sym.sin(ax)],
                         [0,sym.sin(ax),sym.cos(ax)]])
    
        Ry = sym.Matrix([[sym.cos(ay),0,sym.sin(ay)],
                         [0,1,0],
                         [-sym.sin(ay),0,sym.cos(ay)]])
        
        Rz = sym.Matrix([[sym.cos(az),-sym.sin(az),0],
                         [sym.sin(az),sym.cos(az),0],
                         [0,0,1]])
        
        T = (Rz*Ry*Rx).row_join(sym.Matrix([tx,ty,tz])).col_join(
                sym.Matrix([[0,0,0,1]]))
    
        
        # Model: [0, 0, 0, 1]^T = T_W_C * T_P_W * T_I_P * [sx*u, sy*v, 0, 1]^T
        T_W_C = T.subs({tx:self.x1,ty:self.x2,tz:self.x3,az:0,ay:0,ax:0})
    
        # Known transformation T_P_W
        T_P_W = sym.Matrix([[self.c11,self.c12,self.c13,self.c14],
                            [self.c21,self.c22,self.c23,self.c24],
                            [self.c31,self.c32,self.c33,self.c34],
                            [0,0,0,1]])
    
        T_I_P = T.subs({tx:self.x4,ty:self.x5,tz:self.x6,
                        az:self.x7,ay:self.x8,ax:self.x9})
    
        f = T_W_C*T_P_W*T_I_P*sym.Matrix([self.x10*self.u,self.x11*self.v,0,1])
        
        f = f[:3,0]
        Jf = f.jacobian([self.x1,self.x2,self.x3,self.x4,self.x5,self.x6,
                         self.x7,self.x8,self.x9,self.x10,self.x11])
        
        return f, Jf
    
    
    def analyticCalibration(self):
        ######################################################################
        ######################################################################
        # Compute a least squares initial estimation of the US calibration:
        # min||Ax-b|| where
        # A = [ui*R_P_Wi, vi*R_P_Wi, R_P_Wi, -I] is a 3n x 12 matrix (n>=4).
        # x = [sx*R_I_P[:,0], sy*R_I_P[:,1], t_I_P, t_W_C] is a 12 x 1 vector. 
        # and b = [-t_P_W] is a 3n x 1 vector.
        ######################################################################
        ######################################################################
        
        A = []
        b = []
        for ptsi, T_P_Wi in zip(self.pts, self.T_P_W):
            
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
        R_I_P = np.c_[r1, r2, r3] # is not necessarily a rotation matrix. The
                                  # orthonormality constraints were not enforced
        
        # get the closest (Frobenius norm) rotation matrix via SVD
        U, S, Vh = np.linalg.svd(R_I_P)
        R_I_P = U @ Vh.T
        
        # extract the Euler angles
        ax, ay, az = Rotation.from_dcm(R_I_P).as_euler('xyz')
        
        return x[9], x[10], x[11], x[6], x[7], x[8], az, ay, ax, sx, sy
    
    
    def calibEquations(self, f, Jf):
        # Create symbolic equations and symbolic Jacobian matrix
        eq = sym.Matrix()
        Jeq = sym.Matrix()
        
        for ptsi, T_P_Wi in zip(self.pts, self.T_P_W):
            Ri, ti = T_P_Wi[:3,:3], T_P_Wi[:3,-1]
            
            eq = eq.col_join(f.subs({self.u:ptsi[0],self.v:ptsi[1],
                                     self.c11:Ri[0,0],self.c12:Ri[0,1],
                                     self.c13:Ri[0,2],self.c14:ti[0],
                                     self.c21:Ri[1,0],self.c22:Ri[1,1],
                                     self.c23:Ri[1,2],self.c24:ti[1],
                                     self.c31:Ri[2,0],self.c32:Ri[2,1],
                                     self.c33:Ri[2,2],self.c34:ti[2]}))
            
            Jeq = Jeq.col_join(Jf.subs({self.u:ptsi[0],self.v:ptsi[1],
                                       self.c11:Ri[0,0],self.c12:Ri[0,1],
                                       self.c13:Ri[0,2],self.c14:ti[0],
                                       self.c21:Ri[1,0],self.c22:Ri[1,1],
                                       self.c23:Ri[1,2],self.c24:ti[1],
                                       self.c31:Ri[2,0],self.c32:Ri[2,1],
                                       self.c33:Ri[2,2],self.c34:ti[2]}))
        
        return eq, Jeq
    
    
    def iterativeCalibraion(self, eq, Jeq, x0=None):
        ######################################################################
        ######################################################################
        # Compute an iterative least squares solution of the US calibration 
        # with the Levenberg-Marquardt algorithm, in order to refine the above 
        # initial estimation
        ######################################################################
        ######################################################################
        
        
        # Convert model from symbolic to lambda function
        eq = sym.lambdify([self.x1,self.x2,self.x3,self.x4,self.x5,self.x6,
                          self.x7,self.x8,self.x9,self.x10,self.x11], eq)
    
        Jeq = sym.lambdify([self.x1,self.x2,self.x3,self.x4,self.x5,self.x6,
                          self.x7,self.x8,self.x9,self.x10,self.x11], Jeq)
        
        # Objective function that returns function value and jacobian matrix 
        # evaluated in x, which is passed to scipy root function
        def objfunc(x):
            eqi = eq(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10])
            Jeqi = Jeq(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10])
            return eqi.T.tolist()[0], Jeqi
        
        if not x0:
            x0 = self.analyticCalibration()
        
        # Solve problem using Levenberg-Marquardt algorithm
        sol = root(objfunc, x0, jac=True, method='lm',
                   options={'ftol':1e-16,'xtol':1e-16,'gtol':1e-16,
                            'maxiter':5000,'eps':1e-19})
        
        return sol.x.tolist()