import numpy as np
import sympy as sym
from scipy.optimize import root
from scipy.spatial.transform import Rotation


class Calibration:
    '''
    Calibration process handler.
    
    It is possible to create the calibration model, create all equations to
    solve and find an optimal solution using an analytical model or through an
    iterative minimization method with the Levenberg-Marquardt algorithm.
    
    For the transformation matrices, we use the following convention: in
    latex the transformation matrix ^jT_i that represents the
    transformation from i to j, or the transformation matrix of the frame i
    relative to the j frame, it will be written in the code as T_i_j.
    
    Our mathematial model notation in the calibration model is the following:
    [0, 0, 0, 1]^T = T_W_C * T_P_W * T_I_P * [sx*u, sy*v, 0, 1]^T
    where:
    
    -> sx and sy are fixed and represent the scale factor of the ultrasound
    image in x and y directions.
    
    -> u and v are different for each ultrasound image and represent
    the cross-wire ultrasound image coordinates in x and y directions,
    respectively.
    
    -> T_I_P is fixed and represents the transformation from the ultrasound
    image frame {I} to the ultrasound probe frame {P} represented by the target
    of three concentric circles placed in the probe.
    
    -> T_P_W is different for each ultrasound image and represents the
    transformation from the ultrasound probe frame {P} to the world coordinates
    system {W} represented by the stereo vision system (left camera).
    
    -> T_W_C is fixed and represents the transformation from the world
    frame {W} to the cross-wire frame {C} where its origin is located in the
    point of the crossing of wires.
    '''
    
    
    def __init__(self, pts, T_P_W):
        '''
        Init known variables needed in the calibration process.
        
        Also, init symbolic variables used in the calibration equations to
        construct and solve.
        '''
        
        self.setData(pts, T_P_W)
        
        
        # Symbolic variables for known transformation T_P_W
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
        '''
        Set the known variables needed in the calibration problem:
        
        * pts: N x 2 array with the cross-wire image coordinates of the
        ultrasound images.
        * T_P_W: list with the N transformation matrices T_P_W (from the
        probe frame respect to world frame) given by the target pose
        estimation.
        '''
        
        self.pts = pts
        self.T_P_W = T_P_W
    
    
    def T(self, tx, ty, tz, az, ay, ax):
        R = Rotation.from_euler('xyz',[ax,ay,az]).as_dcm()
        
        return np.r_[np.c_[R, [tx,ty,tz]], np.array([[0,0,0,1]])]
        
    
    
    def model(self):
        '''
        Create the three calibration equations symbolically, given by the
        following model:
            
        [0, 0, 0, 1]^T = T_W_C * T_P_W * T_I_P * [sx*u, sy*v, 0, 1]^T,
        
        where for each cross-wire ultrasound image we have the above three
        equiations. Moreover, the Jacobin matrix of these three equations is
        build symbolically.
        
        We use the following convention: in latex the transformation matrix
        ^jT_i that represents the transformation matrix from i to j, or the
        transformation matrix of the frame i with respect to j frame, will be
        write in the code as T_j_i.
        '''
        
        # Variables for rotation matrices and general transformation 
        # matrix T_i_j (^jT_i, i frame respect to j frame).
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
    
        # Known transformation T_P_W (given in the target pose estimation)
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
        '''
        Compute a least squares initial estimation of the US calibration:
        min||Ax-b|| where
        * A = [ui*R_P_Wi, vi*R_P_Wi, R_P_Wi, -I] is a 3n x 12 matrix (n>=4).
        * x = [sx*R_I_P[:,0], sy*R_I_P[:,1], t_I_P, t_W_C] is a 12 x 1 vector. 
        * and b = [-t_P_W] is a 3n x 1 vector.
        '''
        
        # create the A and B matrices
        A = []
        b = []
        for ptsi, T_P_Wi in zip(self.pts, self.T_P_W):
            
            R_P_W, t_P_W = T_P_Wi[:3,:3], T_P_Wi[:3,-1]
        
            A.append(np.c_[ptsi[0]*R_P_W, ptsi[1]*R_P_W, R_P_W, -np.eye(3)])
            b.append(-t_P_W)
        
        
        A = np.vstack(A) # Vertical concatenation of each matrix
        b = np.array(b).flatten()
        x = np.linalg.pinv(A) @ b # Estimation of parameters
        
        # Get scale factors
        sx = np.linalg.norm(x[:3])
        sy = np.linalg.norm(x[3:6])
        
        # Get rotation angles from R_I_P
        r1 = x[:3]/sx
        r2 = x[3:6]/sy
        r3 = np.cross(r1, r2)
        R_I_P = np.c_[r1, r2, r3] # is not necessarily a rotation matrix. The
                                  # orthonormality constraints were not 
                                  # enforced
        
        # get the closest (Frobenius norm) rotation matrix via SVD
        U, S, Vh = np.linalg.svd(R_I_P)
        R_I_P = U @ Vh.T
        
        # extract the Euler angles
        ax, ay, az = Rotation.from_dcm(R_I_P).as_euler('xyz')
        
        return x[9], x[10], x[11], x[6], x[7], x[8], az, ay, ax, sx, sy
    
    
    def calibEquations(self, f, Jf):
        '''
        Based on the three calibration model equations, we create 3 x N
        calibration equations, where N is the number of ultrasound images.
        Here we replace the known variables in the symbolic model and in the
        Jacobian matrix.
        Finally we get a (3 x N) x 1 symbolic matrix eq of equations and a
        (3 x N) x 11 Jacobian matrix Jeq, where 11 is the number of unknows.
        '''
        
        # Create symbolic equations and symbolic Jacobian matrix
        eq = sym.Matrix()
        Jeq = sym.Matrix()
        
        for ptsi, T_P_Wi in zip(self.pts, self.T_P_W):
            Ri, ti = T_P_Wi[:3,:3], T_P_Wi[:3,-1]
            
            # Replace known variables in equations and Jacobian matrix, and
            # add to the respective matrices f and Jeq.
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
        '''
        Compute an iterative least squares solution of the US calibration
        with the Levenberg-Marquardt algorithm, given an initial estimation x0.
        
        We convert the symbolic equation and jacobian matrices to a lambda
        function and build an objective function that return equations and
        Jacobian matrices evaluated in x to implement the Levenberg-Marquardt
        algorithm with the scipy function root.
        
        If an initial estimation x0 is not provided, we use the analytical
        estimation as initialization and refine it.
        '''
        
        
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