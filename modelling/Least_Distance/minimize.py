from scipy.optimize import minimize
from numpy.linalg import norm

def find_min(x, surface):
    """
    Minimizes L2 norm between a fixed point and a surface. x should be an ndarray and surface should be 
    a lambda function that takes one ndarray as input and return 0 if the point is on the surface. 

    returns a scipy.optimize.OptimizeResult object. 

    EXAMPLE USAGE:
        x0 = (0,0,0)
        surface = lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 1 (sphere of radius 1)
        res = find_min(x0, surface)
        print(res.x)
    """
    cons = {'type': 'eq', 'fun': surface}
    return minimize(distance, x, args=(x), constraints=cons)

def distance(x, *args):
    """
    Returns the norm of a point and a fixed point. Formatted as required by 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    return norm(x - args[0])

