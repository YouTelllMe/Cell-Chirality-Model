import numpy as np

from Least_Distance.minimize import find_min
from ModelAB import ModelAB


class FourCell:

    @staticmethod
    def get_velocity(t, y, **kwargs):

        """
        A division by 0 occurs when two cells overlap
        """
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
        k_hat = np.array([0,0,0])

        cortical_flow_r = np.multiply(0.000345*t, np.e**(-0.012732*t))
        cortical_flow_l = np.multiply(0.00071*t, np.e**(-0.0166*t))
        ABal_prime = kwargs['t_final'] * (kwargs['B'] * ((dist12 - 1) * u12 + 
                                        (dist14 - 1) * u14) + 
                                kwargs['A'] * cortical_flow_l * 
                                        (np.cross(-u14, -u34) - 
                                        np.cross(u14, u12)))
        ABar_prime = kwargs['t_final'] * (kwargs['B'] * ((dist12 - 1) * -u12 + 
                                        (dist23 - 1) * u23) + 
                                kwargs['A'] * cortical_flow_r * 
                                        (np.cross(-u23, u34) -
                                        np.cross(u23, -u12)))

        ABpr_prime = kwargs['t_final'] * (kwargs['B'] * ((dist23 - 1) * -u23 + 
                                        (dist34 - 1) * u34) + 
                                kwargs['A'] * cortical_flow_r * 
                                        (np.cross(u23, -u12) -
                                        np.cross(-u23, u34)))

        ABpl_prime = kwargs['t_final'] * (kwargs['B'] * ((dist24 - 1) * -u24 +
                                        (dist34 - 1) * -u34) + 
                                kwargs['A'] * cortical_flow_l * 
                                        (np.cross(u14, u12) -
                                        np.cross(-u14, u34)))
        
        # applies spring force across cells in next iteration
        if dist13 <= 1:
                ABal_prime += kwargs['t_final'] * kwargs['B'] * (dist13 - 1) * u13
                ABpl_prime += kwargs['t_final'] * kwargs['B'] * (dist13 - 1) * -u13

        if dist24 <= 1:
                ABar_prime += kwargs['t_final'] * kwargs['B'] * (dist24 - 1) * u24
                ABpr_prime += kwargs['t_final'] * kwargs['B'] * (dist24 - 1) * -u24

        # cell wall forces 
        for surface in kwargs['surfaces']:
            ABal_prime += kwargs['t_final'] * kwargs['B'] * FourCell._cell_wall_step(ABal, surface)
            ABar_prime += kwargs['t_final'] * kwargs['B'] * FourCell._cell_wall_step(ABar, surface)
            ABpr_prime += kwargs['t_final'] * kwargs['B'] * FourCell._cell_wall_step(ABpr, surface)
            ABpl_prime += kwargs['t_final'] * kwargs['B'] * FourCell._cell_wall_step(ABpl, surface)
            
        return np.concatenate((ABal_prime, ABar_prime, ABpr_prime, ABpl_prime))
    

    @staticmethod
    def _cell_wall_step(pos, surface):
        """
        If the cell is outside the shell the shell pushes it away.

        Current force is Linear. Could also implemet using Van der Waals forces 
        https://en.wikipedia.org/wiki/Van_der_Waals_force
        """

        min_point = find_min(pos, surface)
        distance = np.linalg.norm(min_point.x-pos)
        if distance < 0.5:  
            return np.array((0.5-distance)*(pos-min_point.x)/distance)
        else: 
            return np.zeros(len(pos))