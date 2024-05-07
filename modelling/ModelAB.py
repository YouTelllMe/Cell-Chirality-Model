import numpy as np

class ModelAB:
    
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
        k_hat = np.array([0,0,1])

        cortical_flow_r = np.multiply(0.000345*t, np.e**(-0.012732*t))
        cortical_flow_l = np.multiply(0.00071*t, np.e**(-0.0166*t))
        ABal_prime = kwargs['t_final'] * (kwargs['B'] * ((dist12 - 1) * u12 + 
                                        (dist14 - 1) * u14 - 
                                        (ABal[2] - 0.5) * k_hat) + 
                                kwargs['A'] * cortical_flow_l * 
                                        (np.cross(-u14, -u34) - 
                                        np.cross(u14, u12) -
                                        np.cross(-u12, k_hat)))
        ABar_prime = kwargs['t_final'] * (kwargs['B'] * ((dist12 - 1) * -u12 + 
                                        (dist23 - 1) * u23 - 
                                        (ABar[2] - 0.5) * k_hat) + 
                                kwargs['A'] * cortical_flow_r * 
                                        (np.cross(-u23, u34) -
                                        np.cross(u23, -u12) -
                                        np.cross(u12, k_hat)))

        ABpr_prime = kwargs['t_final'] * (kwargs['B'] * ((dist23 - 1) * -u23 + 
                                        (dist34 - 1) * u34 - 
                                        (ABpr[2] - 0.5) * k_hat) + 
                                kwargs['A'] * cortical_flow_r * 
                                        (np.cross(u23, -u12) -
                                        np.cross(-u23, u34) -
                                        np.cross(-u34, k_hat)))

        ABpl_prime = kwargs['t_final'] * (kwargs['B'] * ((dist14 - 1) * -u14 +
                                        (dist34 - 1) * -u34 - 
                                        (ABpl[2] - 0.5) * k_hat) + 
                                kwargs['A'] * cortical_flow_l * 
                                        (np.cross(u14, u12) -
                                        np.cross(-u14, u34) -
                                        np.cross(u34, k_hat)))
        
        # applies spring force across cells in next iteration
        if dist13 <= 1:
                ABal_prime += kwargs['t_final'] * kwargs['B'] * (dist13 - 1) * u13
                ABpr_prime += kwargs['t_final'] * kwargs['B'] * (dist13 - 1) * -u13

        if dist24 <= 1:
                ABar_prime += kwargs['t_final'] * kwargs['B'] * (dist24 - 1) * u24
                ABpl_prime += kwargs['t_final'] * kwargs['B'] * (dist24 - 1) * -u24

        return np.concatenate((ABal_prime, 
                               ABar_prime, 
                               ABpr_prime, 
                               ABpl_prime),
                               axis=None)
        