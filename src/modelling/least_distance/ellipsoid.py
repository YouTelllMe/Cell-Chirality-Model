import numpy as np

def get_ellipse_root():
    """
    Root finding for ellipse using the Bisection Method. 
    https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
    """
    pass


def min_point_prolate(y: tuple[float, float, float], e0: float, e1: float) -> tuple[float, float, float]:
    """
    Finds distance from a point in R3 to an ellipsoid. Assumes the ellipsoid is 
    a Prolate Spheriod where e0 > e1 = e2 for the ellipsoid with equation 
        y^2/e0^2 + (x1^2+x2^2)/e1^2 
    https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf


    we let x = y[0], y = y[1], z = y[2] since the x axis separates the anterior and posterior. 

    requires e0 >= e1 > 0
    """    
    #change of variables to 2 dimensional problem; we interpret the Prolate Spheriod as a 2D ellipse
    r = np.sqrt(y[1]**2+y[2]**2)
    y0 = abs(y[0])

    if r > 0:
        if y0 > 0:
            t_bar = get_ellipse_root()
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
            





