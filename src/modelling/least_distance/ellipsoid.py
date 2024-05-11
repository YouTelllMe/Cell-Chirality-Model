import numpy as np

def get_ellipse_root(y0, y1, e0, e1, maxit = 10**4):
    """
    Root finding for ellipse using the Bisection Method. 
    https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
    """
    z0 = y0/e0
    z1 = y1/e1
    r0 = np.sqrt(e0/e1)
    n0 = r0 * z0
    s0 = z1 - 1
    # inside the ellipse
    if (y0**2/e0**2 + y1**2/e1**2 - 1) < 0:
        s1 = 0
    else:
        s1 = np.norm([n0, z1]) - 1
    s = 0
    
    iteration = 0
    while iteration < maxit:
        s = (s0+s1)/2
        if s == s0 or s == s1:
            return s
        
        ratio0 = n0/(s+r0)
        ratio1 = z1/(s+1)
        g = np.sqrt(ratio0) + np.sqrt(ratio1) - 1
        if g > 0:
            s0 = s
        elif g < 0:
            s1 = s
        else:
            return s


def min_point_prolate(y: tuple[float, float, float], e0: float, e1: float) -> tuple[float, float, float]:
    """
    Finds distance from a point in R3 to an ellipsoid. Assumes the ellipsoid is 
    a Prolate Spheriod where e0 > e1 = e2 for the ellipsoid with equation 
        x0^2/e0^2 + (x1^2+x2^2)/e1^2 
    https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf


    we let x = x0, y = x1, z = x2 since the x axis separates the anterior and posterior. 

    requires e0 >= e1 > 0
    """    

    # on ellipse
    if (y[0]**2/e0**2 + y[0]**2/e1**2 + y[0]**2/e1**2 - 1) == 0:
        return np.array(y)

    #change of variables to 2 dimensional problem; we interpret the Prolate Spheriod as a 2D ellipse
    y0 = abs(y[0])
    r = np.sqrt(y[1]**2+y[2]**2)

    if r > 0:
        if y0 > 0:
            t_bar = get_ellipse_root(y0, r, e0, e1)
            return np.array([e0*e0*y0 / (t_bar+e0*e0), e1*e1*r / (t_bar+e1*e1)])
        else:
            return np.array([0, e1])
    else:
        if y0 < ((e0*e0 - e1*e1)/e0):
            x0 = e0*e0*y0/(e0*e0-e1*e1)
            x1 = e1*np.sqrt(1-(x0/e0)*(x0/e0))
            return np.array([x0, x1])
        else:
            return np.array([e0, 0])
            





