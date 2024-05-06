import numpy as np

from Least_Distance.minimize import find_min
from ModelAB import ModelAB


class FourCell:

    @staticmethod
    def get_velocity(t, y, **kwargs):

        ABal = np.array([y[0], y[1], y[2]])
        ABar = np.array([y[3], y[4], y[5]])
        ABpr = np.array([y[6], y[7], y[8]])
        ABpl = np.array([y[9], y[10], y[11]])

        AB_prime = ModelAB.get_velocity(t, y, A=kwargs['A'], B=kwargs['B'])
        ABal_prime = np.array([AB_prime[0], AB_prime[1], AB_prime[2]])
        ABar_prime = np.array([AB_prime[3], AB_prime[4], AB_prime[5]])
        ABpr_prime = np.array([AB_prime[6], AB_prime[7], AB_prime[8]])
        ABpl_prime = np.array([AB_prime[9], AB_prime[10], AB_prime[11]])

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