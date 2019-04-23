import sympy as sym
import pickle

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


u, v = sym.symbols('u v')

x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = \
sym.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11')

c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34 = \
sym.symbols('c11 c12 c13 c14 c21 c22 c23 c24 c31 c32 c33 c34')

Tc = sym.Matrix([[c11,c12,c13,c14],[c21,c22,c23,c24],[c31,c32,c33,c34],
                 [0,0,0,1]])

f = T.subs({tx:x9,ty:x10,tz:x11,az:0,ay:0,ax:0})*Tc*T.subs({tx:x3,ty:x4,tz:x5,
          az:x6,ay:x7,ax:x8})*sym.Matrix([x1*u,x2*v,0,1])
f = f[:3,:]

Jf = sym.simplify(f.jacobian([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]))

with open('models.pkl', 'wb') as file:
    pickle.dump(f, file)
    pickle.dump(Jf, file)



'''import sympy as sym
from scipy.optimize import root


v0, v1 = sym.symbols('v[0] v[1]')

f = sym.Matrix([v0 * sym.cos(v1) - 4, v1*v0 - v1- 5])
Jf = f.jacobian(sym.Matrix([v0,v1]))

flmb = sym.lambdify([v0,v1], f)
Jflmb = sym.lambdify([v0,v1], Jf)

def func2(x):
    return flmb(x[0],x[1]).T.tolist()[0], Jflmb(x[0],x[1])

sol = root(func2, [1, 1], jac=True, method='lm')
print(sol.x)'''
