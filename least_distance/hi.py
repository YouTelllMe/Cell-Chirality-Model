from typing import Callable

from sympy import Lambda, symbols, diff, solve, sqrt, Eq, lambdify
from sympy.core.symbol import Symbol



def minimize3DDistance(
        f: Lambda, 
        g: Lambda,
        h: Lambda):
    """
    Use Lagrange Multipliers to find the points that yields 

    REQUIRES
    - f, g, h functions that depend on symbols x1 y1 z1 x2 y2 z2
    """
    fx1, fy1, fz1, fx2, fy2, fz2 = gradient6D(f)
    gx1, gy1, gz1, gx2, gy2, gz2 = gradient6D(g)
    hx1, hy1, hz1, hx2, hy2, hz2 = gradient6D(h)

    sols = solve([fx1(*p)-alpha * gx1(*p)- beta * hx1(*p),
                  fy1(*p)-alpha * gy1(*p)- beta * hy1(*p),
                  fz1(*p)-alpha * gz1(*p)- beta * hz1(*p),
                  fx2(*p)-alpha * gx2(*p)- beta * hx2(*p),
                  fy2(*p)-alpha * gy2(*p)- beta * hy2(*p),
                  fz2(*p)-alpha * gz2(*p)- beta * hz2(*p),
                  g(*p),
                  h(*p)
                  ], x1, y1, z1, x2, y2, z2, alpha, beta)
        
    return sols



def gradient6D(f: Lambda):
    """
    """
    gradient = []

    for var in list(p):
        gradient.append(lambdify(p, diff(f(*p), var)))
    
    return gradient

def generate_symbols():
    """
    """
    x1, y1, z1, x2, y2, z2, alpha, beta = symbols('x1 y1 z1 x2 y2 z2 alpha beta')
    return {
        "x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2":  z2, "alpha": alpha, "beta": beta
    }


x1, y1, z1, x2, y2, z2, alpha, beta = symbols('x1 y1 z1 x2 y2 z2 alpha beta')
p = x1, y1, z1, x2, y2, z2
# print(*p)
euclidean3D = Lambda((x1, y1, z1, x2, y2, z2), (x1-x2)**2
                                                    +(y1-y2)**2
                                                    +(z1-z2)**2)
sphere1 = Lambda((x1, y1, z1, x2, y2, z2), x1**2 + y1**2 + z1 ** 2 - 1)
# sphere2 = Lambda((x1, y1, z1, x2, y2, z2), x2**2 + y2**2 + z2 ** 2 - 10)
a = 10
b = 50
c = 100
ellipsoid = Lambda((x1, y1, z1, x2, y2, z2), (x2**2)/(a**2) + (y2**2)/(b**2) + (z2 ** 2)/(c**2) - 1)

for i in minimize3DDistance(euclidean3D, sphere1, ellipsoid):
    print(i, "\n")