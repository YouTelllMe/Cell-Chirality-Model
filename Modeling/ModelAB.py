from Modeling.Cell import Cell, FourCellSystem
import numpy as np
import config

class ModelAB:
    
    @staticmethod
    def get_velocity(**kwargs):
        t_final = kwargs.t_final
        ABal = np.array([kwargs.cur_pos[0], kwargs.cur_pos[1], kwargs.cur_pos[2]])
        ABar = np.array([kwargs.cur_pos[3], kwargs.cur_pos[4], kwargs.cur_pos[5]])
        ABpr = np.array([kwargs.cur_pos[6], kwargs.cur_pos[7], kwargs.cur_pos[8]])
        ABpl = np.array([kwargs.cur_pos[9], kwargs.cur_pos[10], kwargs.cur_pos[11]])

        dist12 = np.linalg.norm(ABal-ABar)
        dist13 = np.linalg.norm(ABal-ABpr)
        dist14 = np.linalg.norm(ABal-ABpl)
        dist23 = np.linalg.norm(ABar-ABpr)
        dist24 = np.linalg.norm(ABar-ABpl)
        dist34 = np.linalg.norm(ABpr-ABpl)

        u12 = (ABal-ABar) / dist12 # unit vector from 2 to 1 
        u13 = (ABal-ABpr) / dist13 # unit vector from 3 to 1
        u14 = (ABal-ABpl) / dist14 # unit vector from 4 to 1
        u23 = (ABar-ABpr) / dist23 # unit vector from 3 to 2
        u24 = (ABar-ABpl) / dist24 # unit vector from 4 to 2
        u34 = (ABpr-ABpl) / dist34 # unit vector from 4 to 3
        k_hat = np.array([0,0,0])

        cortical_flow_r = np.multiply(0.000345*kwargs.tau, np.e**(-0.012732*kwargs.tau))
        cortical_flow_l = np.multiply(0.00071*kwargs.tau, np.e**(-0.0166*kwargs.tau))

        ABal_prime = t_final * (kwargs.B * ((dist12 - 1) * u12 + 
                                        (dist13 - 1) * u13 - 
                                        (ABal[2] - 0.5) * k_hat) + 
                                kwargs.A * cortical_flow_r * 
                                        (np.cross(-u12, u24) - 
                                        np.cross(u12, u13) -
                                        np.cross(u13, k_hat)))
        ABar_prime = t_final * (kwargs.B * ((dist12 - 1) * -u12 + 
                                        (dist24 - 1) * u24 - 
                                        (ABar[2] - 0.5) * k_hat) + 
                                kwargs.A * cortical_flow_r * 
                                        (np.cross(u12, u13) -
                                        np.cross(-u12, u24) -
                                        np.cross(u24, k_hat)))

        ABpr_prime = t_final * (kwargs.B * ((dist13 - 1) * -u13 + 
                                        (dist34 - 1) * u34 - 
                                        (ABpr[2] - 0.5) * k_hat) + 
                                kwargs.A * cortical_flow_l * 
                                        (np.cross(-u34, -u24) -
                                        np.cross(u34, -u13) -
                                        np.cross(-u13, k_hat)))

        ABpl_prime = t_final * (kwargs.B * ((dist24 - 1) * -u24 +
                                        (dist34 - 1) * -u34 - 
                                        (ABpl[2] - 0.5) * k_hat) + 
                                kwargs.A * cortical_flow_l * 
                                        (np.cross(u34, -u13) -
                                        np.cross(u34, u24) -
                                        np.cross(-u24, k_hat)))
        
        # applies spring force across cells in next iteration
        if dist13 <= 1:
                ABal_prime += t_final * kwargs.B * (dist13 - 1) * u13
                ABpl_prime += t_final * kwargs.B * (dist13 - 1) * -u13

        if dist24 <= 1:
                ABar_prime += t_final * kwargs.B * (dist24 - 1) * u24
                ABpr_prime += t_final * kwargs.B * (dist24 - 1) * -u24

        return np.concatenate((ABal_prime, 
                               ABar_prime, 
                               ABpr_prime, 
                               ABpl_prime),
                               axis=None)
        