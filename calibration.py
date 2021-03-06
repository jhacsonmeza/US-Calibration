import io
import itertools
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
    "latex notation" the transformation matrix ^jT_i that represents the
    transformation from i to j, or that describe the frame i relative to the
    frame j, it will be written in the code as T_i_j.
    
    Our mathematial notation in the calibration model is the following:
    [0, 0, 0, 1]^T = T_W_F * T_T_W * T_I_T * [sx*u, sy*v, 0, 1]^T
    where:
    
    -> sx and sy are fixed and represent the scale factor of the ultrasound
    image in x and y directions.
    
    -> u and v are different for each ultrasound image and represent
    the cross-wire ultrasound image coordinates in x and y directions,
    respectively.
    
    -> T_I_T is fixed and represents the transformation from the ultrasound
    image frame {I} to the ultrasound transducer frame {T} represented by the 
    target of the three coplanar circles placed in the probe.
    
    -> T_T_W is different for each ultrasound image and represents the
    transformation from the transducer frame {T} to the world coordinates
    system {W} represented by the stereo vision system (left camera).
    
    -> T_W_F is fixed and represents the transformation from the world
    frame {W} to the cross-wire phatom frame {F} where its origin is located
    in the point of the crossing of wires.
    '''
    
    
    def __init__(self, pts=None, T_T_W=None):
        '''
        Init known variables needed in the calibration process if are
        provided in the construction of the object. By default are None.
        
        Also, init symbolic variables used in the calibration equations to
        construct and solve them.
        '''
        
        # If provided, init pts and T_T_W
        if pts is not None and T_T_W is not None:
            self.setData(pts, T_T_W)
        
        
        # Symbolic variables for known transformation T_T_W
        self.c11, self.c12, self.c13, self.c14 = sym.symbols('c11 c12 c13 c14')
        self.c21, self.c22, self.c23, self.c24 = sym.symbols('c21 c22 c23 c24')
        self.c31, self.c32, self.c33, self.c34 = sym.symbols('c31 c32 c33 c34')
        
        # Symbolic variables for image coordinates of point phantom
        self.u, self.v = sym.symbols('u v')
        
        # Symbolic variables for unknowns:
        # from T_W_F: tx -> x1, ty -> x2, tz -> x3
        # from T_I_T: tx -> x4, ty -> x5, tz -> x6, az -> x7, ay -> x8, ax -> x9
        # sx -> x10, sy -> x11
        self.x1, self.x2, self.x3 = sym.symbols('x1 x2 x3')
        self.x4, self.x5, self.x6 = sym.symbols('x4 x5 x6')
        self.x7, self.x8, self.x9 = sym.symbols('x7 x8 x9')
        self.x10, self.x11 = sym.symbols('x10 x11')
    
    
    def setData(self, pts, T_T_W):
        '''
        Set the known variables needed in the calibration problem:
        
        * pts: N x 2 array with the cross-wire image coordinates of the
        ultrasound images.
        * T_T_W: N x 4 x 4 array with the N transformation matrices T_T_W 
        (which describe the probe frame relative to the world frame) given by
        the target pose estimation.
        '''
        
        self.pts = pts
        self.T_T_W = T_T_W
    
    
    def T(self, tx, ty, tz, az, ay, ax):
        R = Rotation.from_euler('xyz',[ax,ay,az]).as_dcm()
        
        return np.r_[np.c_[R, [tx,ty,tz]], [[0,0,0,1]]]
        
    
    
    def model(self):
        '''
        Create the three calibration equations symbolically, given by the
        following model:
            
        [0, 0, 0, 1]^T = T_W_F * T_T_W * T_I_T * [sx*u, sy*v, 0, 1]^T,
        
        where for each cross-wire ultrasound image we have the above three
        equiations. Moreover, the Jacobin matrix of these three equations is
        build symbolically.
        
        We use the following convention: in latex the transformation matrix
        ^jT_i that represents the transformation matrix from i to j, or the
        transformation matrix of the frame i with respect to j frame, will be
        write in the code as T_j_i.
        '''
        
        # Variables for rotation matrices and general transformation 
        # matrix T_i_j (^jT_i, frame i relative to frame j).
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
    
        
        # Model: [0, 0, 0, 1]^T = T_W_F * T_T_W * T_I_T * [sx*u, sy*v, 0, 1]^T
        T_W_F = T.subs({tx:self.x1,ty:self.x2,tz:self.x3,az:0,ay:0,ax:0})
    
        # Known transformation T_T_W (given in the target pose estimation)
        T_T_W = sym.Matrix([[self.c11,self.c12,self.c13,self.c14],
                            [self.c21,self.c22,self.c23,self.c24],
                            [self.c31,self.c32,self.c33,self.c34],
                            [0,0,0,1]])
        
        T_I_T = T.subs({tx:self.x4,ty:self.x5,tz:self.x6,
                        az:self.x7,ay:self.x8,ax:self.x9})
        
        f = T_W_F*T_T_W*T_I_T*sym.Matrix([self.x10*self.u,self.x11*self.v,0,1])
        
        f = f[:3,0]
        Jf = f.jacobian([self.x1,self.x2,self.x3,self.x4,self.x5,self.x6,
                         self.x7,self.x8,self.x9,self.x10,self.x11])
        
        return f, Jf
    
    
    def analyticCalibration(self):
        '''
        Compute a least squares initial estimation of the US calibration:
        min||Ax-b|| where
        * A = [ui*R_T_Wi, vi*R_T_Wi, R_T_Wi, I] is a 3n x 12 matrix (n>=4).
        * x = [sx*R_I_T[:,0], sy*R_I_T[:,1], t_I_T, t_W_F] is a 12 x 1 vector.
        * and b = [-t_T_W] is a 3n x 1 vector.
        
        Based on analyticLeastSquaresEstimate function from:
        https://github.com/zivy/LSQRRecipes/blob/master/parametersEstimators/
        SinglePointTargetUSCalibrationParametersEstimator.cxx#L120
        '''
        
        # create the A and b matrices
        A = []
        b = []
        for ptsi, T_T_Wi in zip(self.pts, self.T_T_W):
            
            R_T_W, t_T_W = T_T_Wi[:3,:3], T_T_Wi[:3,-1]
        
            A.append(np.c_[ptsi[0]*R_T_W, ptsi[1]*R_T_W, R_T_W, np.eye(3)])
            b.append(-t_T_W)
        
        
        A = np.vstack(A) # Vertical concatenation of each matrix
        b = np.array(b).flatten()
        x = np.linalg.pinv(A) @ b # Estimation of parameters
        
        # Get scale factors
        sx = np.linalg.norm(x[:3])
        sy = np.linalg.norm(x[3:6])
        
        # Get rotation angles from R_I_T
        r1 = x[:3]/sx
        r2 = x[3:6]/sy
        r3 = np.cross(r1, r2)
        R_I_T = np.c_[r1, r2, r3] # is not necessarily a rotation matrix. The
                                  # orthonormality constraints were not 
                                  # enforced
        
        # get the closest (Frobenius norm) rotation matrix via SVD
        U, S, Vh = np.linalg.svd(R_I_T)
        R_I_T = U @ Vh.T
        
        # extract the Euler angles
        ax, ay, az = Rotation.from_dcm(R_I_T).as_euler('xyz')
        
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
        
        for ptsi, T_T_Wi in zip(self.pts, self.T_T_W):
            Ri, ti = T_T_Wi[:3,:3], T_T_Wi[:3,-1]
            
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
    
        # Measure RMS error
        fmin = eq(sol.x[0],sol.x[1],sol.x[2],sol.x[3],sol.x[4],sol.x[5],
                  sol.x[6],sol.x[7],sol.x[8],sol.x[9],sol.x[10])
        rms = np.sqrt(np.mean(np.array(fmin)**2))
        
        return sol.x.tolist(), rms
    
    
    def calibEvaluation(self, rms, rc, rtr, rtl, rbr, rbl):
        n = rc.shape[0] # Number of calibrations
        yield '-> Results for a total of {} calibrations:'.format(n)
        
        
        # Report mean RMS error of equations
        yield '\n\nMean RMS error of equations = {} mm'.format(rms.mean())
        
        
        # Calculate calibration reproducibility precision at the five points
        errc = np.linalg.norm(rc-rc.mean(0),axis=1)
        errtr = np.linalg.norm(rtr-rtr.mean(0),axis=1)
        errtl = np.linalg.norm(rtl-rtl.mean(0),axis=1)
        errbr = np.linalg.norm(rbr-rbr.mean(0),axis=1)
        errbl = np.linalg.norm(rbl-rbl.mean(0),axis=1)
        
        mu_CR1_mean = np.r_[errc,errtr,errtl,errbr,errbl].mean()
        yield '\n\n\u03BC_CR1 at center = {} mm'.format(errc.mean())
        yield '\n\u03BC_CR1 at bottom right = {} mm'.format(errbr.mean())
        yield '\n\u03BC_CR1 mean = {} mm'.format(mu_CR1_mean)
        
        
        # Precision with all possible pairs of calibrations in each point
        errc = np.array([np.linalg.norm(p1-p2) for p1, p2 in 
                          itertools.combinations(rc,2)])
        errtr = np.array([np.linalg.norm(p1-p2) for p1, p2 in 
                           itertools.combinations(rtr,2)])
        errtl = np.array([np.linalg.norm(p1-p2) for p1, p2 in 
                           itertools.combinations(rtl,2)])
        errbr = np.array([np.linalg.norm(p1-p2) for p1, p2 in 
                           itertools.combinations(rbr,2)])
        errbl = np.array([np.linalg.norm(p1-p2) for p1, p2 in 
                           itertools.combinations(rbl,2)])
        
        mu_CR2_mean = np.r_[errc, errtr, errtl, errbr, errbl].mean()
        yield '\n\n\u03BC_CR2 at center = {} mm'.format(errc.mean())
        yield '\n\u03BC_CR2 at bottom right corner = {} mm'.format(errbr.mean())
        yield '\n\u03BC_CR2 mean = {} mm'.format(mu_CR2_mean)


    def writeReport(self, path, rms, rc, rtr, rtl, rbr, rbl):
        with io.open(path, 'w', encoding='utf-8') as fp:
            lines = self.calibEvaluation(rms, rc, rtr, rtl, rbr, rbl)
            fp.writelines(lines)