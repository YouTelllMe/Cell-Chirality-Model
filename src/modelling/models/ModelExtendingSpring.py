import numpy as np
from modelling.least_distance.minimize import find_min
from modelling.least_distance.ellipsoid import min_point_ellpsoid
import time

def get_velocity(A, B, t_final):

    def func(t, y):
        """
        A division by 0 occurs when two cells overlap
        """
        # start = time.time()
        ABal = np.array([y[0], y[1], y[2]])
        ABar = np.array([y[3], y[4], y[5]])
        ABpr = np.array([y[6], y[7], y[8]])
        ABpl = np.array([y[9], y[10], y[11]])

        dist12 = np.linalg.norm(ABal-ABar)
        dist13 = np.linalg.norm(ABal-ABpr)
        dist14 = np.linalg.norm(ABal-ABpl)
        dist23 = np.linalg.norm(ABar-ABpr)
        dist24 = np.linalg.norm(ABar-ABpl)
        dist34 = np.linalg.norm(ABpr-ABpl)

        u12 = (ABar-ABal) / dist12 # 2-1 
        u13 = (ABpr-ABal) / dist13 # 3-1 
        u14 = (ABpl-ABal) / dist14 # 4-1
        u23 = (ABpr-ABar) / dist23 # 3-2
        u24 = (ABpl-ABar) / dist24 # 4-2
        u34 = (ABpl-ABpr) / dist34 # 4-3

        cortical_flow_r = 0.000345*t*t_final*np.e**(-0.012732*t*t_final)
        cortical_flow_l = 0.00071*t*t_final*np.e**(-0.0166*t*t_final)

        # avg of two cortical flows
        a = 0.0005275 
        lam = 0.014666
        cortical_int = a*t*t_final*(-t/(lam * t_final)*np.e**(-lam * t_final * t)-1/(lam**2*t_final**2)*np.e**(-lam * t_final*t)+1/(lam**2*t_final**2))
        cortical_int_scale = 51.040149469200486 # ensures at the end, final spring length is 1.5
        cortical_int *= cortical_int_scale

        ABal_prime = t_final * (B * ((dist12 - (1 + cortical_int)) * u12 + 
                                        (dist14 - (1 + cortical_int)) * u14) + 
                                A * cortical_flow_l * 
                                        (np.cross(-u14, -u34) - 
                                        np.cross(u14, u12)))
        ABar_prime = t_final * (B * ((dist12 - (1 + cortical_int)) * -u12 + 
                                        (dist23 - (1 + cortical_int)) * u23) + 
                                A * cortical_flow_r * 
                                        (np.cross(-u23, u34) -
                                        np.cross(u23, -u12)))

        ABpr_prime = t_final * (B * ((dist23 - (1 + cortical_int)) * -u23 + 
                                        (dist34 - (1 + cortical_int)) * u34) + 
                                A * cortical_flow_r * 
                                        (np.cross(u23, -u12) -
                                        np.cross(-u23, u34)))

        ABpl_prime = t_final * (B * ((dist14 - (1 + cortical_int)) * -u14 +
                                        (dist34 - (1 + cortical_int)) * -u34) + 
                                A * cortical_flow_l * 
                                        (np.cross(u14, u12) -
                                        np.cross(-u14, -u34)))
        
        # applies spring force across cells in next iteration
        if dist13 <= 1:
                ABal_prime += t_final * B * (dist13 - 1) * u13
                ABpr_prime += t_final * B * (dist13 - 1) * -u13

        if dist24 <= 1:
                ABar_prime += t_final * B * (dist24 - 1) * u24
                ABpl_prime += t_final * B * (dist24 - 1) * -u24

        # cell wall forces 
        ABal_prime += t_final * B * _cell_wall_step(ABal)
        ABar_prime += t_final * B * _cell_wall_step(ABar)
        ABpr_prime += t_final * B * _cell_wall_step(ABpr)
        ABpl_prime += t_final * B * _cell_wall_step(ABpl)

        # print("time(s): ", time.time()-start)    

        return np.concatenate((ABal_prime, ABar_prime, ABpr_prime, ABpl_prime))
    return func


def _cell_wall_step(pos):
        """
        If the cell is outside the shell the shell pushes it away.

        Current force is Linear. Could also implemet using Van der Waals forces 
        https://en.wikipedia.org/wiki/Van_der_Waals_force
        """

        e0, e1 = (2.5*1.5, 2.5)
        min_point = min_point_ellpsoid(pos, e0, e1)
        distance = np.linalg.norm(min_point-pos)
        if distance < 0.5:  
                return np.array((0.5-distance)*(pos-min_point)/distance)
        else: 
                return np.zeros(len(pos))