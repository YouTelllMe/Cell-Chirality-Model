from Modeling.Cell import Cell, FourCellSystem
import numpy as np
import config

class ModelAB():
    def __init__(self, A, B, cur_pos) -> None:
        self.A = A
        self.B = B
        self.cur_pos = cur_pos

    def get_velocity(self, time):
        t_final = config.T_FINAL
        # position vectors
        ABal = np.array([self.cur_pos[0], self.cur_pos[1], self.cur_pos[2]])
        ABar = np.array([self.cur_pos[3], self.cur_pos[4], self.cur_pos[5]])
        ABpr = np.array([self.cur_pos[6], self.cur_pos[7], self.cur_pos[8]])
        ABpl = np.array([self.cur_pos[9], self.cur_pos[10], self.cur_pos[11]])

        distance12 = np.linalg.norm(ABal-ABar)
        distance13 = np.linalg.norm(ABal-ABpr)
        distance14 = np.linalg.norm(ABal-ABpl)
        distance23 = np.linalg.norm(ABar-ABpr)
        distance24 = np.linalg.norm(ABar-ABpl)
        distance34 = np.linalg.norm(ABpr-ABpl)

        u12 = (ABal-ABar) / distance12 # unit vector from 2 to 1 
        u13 = (ABal-ABpr) / distance13 # unit vector from 3 to 1
        u14 = (ABal-ABpl) / distance14 # unit vector from 4 to 1
        u23 = (ABar-ABpr) / distance23 # unit vector from 3 to 2
        u24 = (ABar-ABpl) / distance24 # unit vector from 4 to 2
        u34 = (ABpr-ABpl) / distance34 # unit vector from 4 to 3
        k_hat = np.array([0,0,0])


        # cortical_flow = np.multiply(0.000527*time, np.e**(-0.01466569*time))
        cortical_flow_r = np.multiply(0.000345*time, np.e**(-0.012732*time))
        cortical_flow_l = np.multiply(0.00071*time, np.e**(-0.0166*time))

        ABal_prime = t_final * (self.B * ((distance12 - 1) * u12 + 
                                        (distance13 - 1) * u13 - 
                                        (ABal[2] - 0.5) * k_hat) + 
                                self.A * cortical_flow_r * 
                                        (np.cross(-u12, u24) - 
                                        np.cross(u12, u13) -
                                        np.cross(u13, k_hat)))
        ABar_prime = t_final * (self.B * ((distance12 - 1) * -u12 + 
                                        (distance24 - 1) * u24 - 
                                        (ABar[2] - 0.5) * k_hat) + 
                                self.A * cortical_flow_r * 
                                        (np.cross(u12, u13) -
                                        np.cross(-u12, u24) -
                                        np.cross(u24, k_hat)))

        ABpr_prime = t_final * (self.B * ((distance13 - 1) * -u13 + 
                                        (distance34 - 1) * u34 - 
                                        (ABpr[2] - 0.5) * k_hat) + 
                                self.A * cortical_flow_l * 
                                        (np.cross(-u34, -u24) -
                                        np.cross(u34, -u13) -
                                        np.cross(-u13, k_hat)))

        ABpl_prime = t_final * (self.B * ((distance24 - 1) * -u24 +
                                        (distance34 - 1) * -u34 - 
                                        (ABpl[2] - 0.5) * k_hat) + 
                                self.A * cortical_flow_l * 
                                        (np.cross(u34, -u13) -
                                        np.cross(u34, u24) -
                                        np.cross(-u24, k_hat)))
        
        # applies spring force across cells in next iteration
        if self.system.info.dist_14 <= 1:
                ABal_prime += t_final * self.B * (distance14 - 1) * u14
                ABpl_prime += t_final * self.B * (distance14 - 1) * -u14

        if self.system.info.dist_23 <= 1:
                ABar_prime += t_final * self.B * (distance23 - 1) * u23
                ABpr_prime += t_final * self.B * (distance23 - 1) * -u23

        return np.concatenate((ABal_prime, 
                               ABar_prime, 
                               ABpr_prime, 
                               ABpl_prime),
                               axis=None)
        