import pickle
import sympy as sym

base = 'Calibration test 11-04-19/'


# Variables for rotation matrices and general transformation matrix T_i_j
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


# Variables for known transformation T_p_w
c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34 = \
sym.symbols('c11 c12 c13 c14 c21 c22 c23 c24 c31 c32 c33 c34')

T_p_w = sym.Matrix([[c11,c12,c13,c14],[c21,c22,c23,c24],[c31,c32,c33,c34],
                 [0,0,0,1]])


# Symbolic variables for image coordinates of point phantom
u, v = sym.symbols('u v')

# Symbolic variables for unknowns:
# from T_W_C: tx -> x1, ty -> x2, tz -> x3
# from T_I_P: tx -> x4, ty -> x5, tz -> x6, az -> x7, ay -> x8, ax -> x9
# sx -> x10, sy -> x11
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = \
sym.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11')

# Model: [0, 0, 0, 1]^T = T_W_C * T_P_W* T_I_P * [sx*u, sy*v, 0, 1]^T
f = T.subs({tx:x1,ty:x2,tz:x3,az:0,ay:0,ax:0})*T_p_w*T.subs({tx:x4,ty:x5,
          tz:x6,az:x7,ay:x8,ax:x9})*sym.Matrix([x10*u,x11*v,0,1])

f = sym.Matrix([sym.sqrt(f[0,0]**2+f[1,0]**2+f[2,0]**2)])
Jf = f.jacobian([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])
#f = f[:3,0]
#Jf = f.jacobian([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])



# load known variables
with open(base+'known_variables.pkl','rb') as file:
    pts = pickle.load(file)
    T_P_W = pickle.load(file)


# Create symbolic equations and symbolic Jacobian matrix
eq = sym.Matrix()
Jeq = sym.Matrix()
for ptsi, T_P_Wi in zip(pts, T_P_W):
    Rm, tvec = T_P_Wi[:3,:3], T_P_Wi[:3,-1]
    
    eq = eq.col_join(f.subs({u:ptsi[0],v:ptsi[1],c11:Rm[0,0],
                             c12:Rm[0,1],c13:Rm[0,2],c14:tvec[0],
                             c21:Rm[1,0],c22:Rm[1,1],c23:Rm[1,2],
                             c24:tvec[1],c31:Rm[2,0],c32:Rm[2,1],
                             c33:Rm[2,2],c34:tvec[2]}))
    
    Jeq = Jeq.col_join(Jf.subs({u:ptsi[0],v:ptsi[1],c11:Rm[0,0],
                             c12:Rm[0,1],c13:Rm[0,2],c14:tvec[0],
                             c21:Rm[1,0],c22:Rm[1,1],c23:Rm[1,2],
                             c24:tvec[1],c31:Rm[2,0],c32:Rm[2,1],
                             c33:Rm[2,2],c34:tvec[2]}))
        

    
# Save model
with open(base+'model.pkl', 'wb') as file:
    pickle.dump(eq, file)
    pickle.dump(Jeq, file)