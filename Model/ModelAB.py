from Cell import Cell, FourCellSystem
import numpy as np
import config
from Model.Model import Model

class ModelAB(Model):
    def __init__(self, A, B, system) -> None:
        super().__init__(A, B, system)

    def step(self, time):
        super().step(time)
        t_final = config.T_FINAL
        # position vectors
        p1 = self.system.p1.get_position()
        p2 = self.system.p2.get_position()
        p3 = self.system.p3.get_position()
        p4 = self.system.p4.get_position()

        p1z = p1[2]
        p2z = p2[2]
        p3z = p3[2]
        p4z = p4[2]
        k_hat = np.array([0,0,1])

        # unit vectors 
        u12 = np.subtract(p2, p1)/np.linalg.norm(np.subtract(p2, p1))
        u13 = np.subtract(p3, p1)/np.linalg.norm(np.subtract(p3, p1))
        u14 = np.subtract(p4, p1)/np.linalg.norm(np.subtract(p4, p1))
        u21 = -1 * u12
        u23 = np.subtract(p3, p2)/np.linalg.norm(np.subtract(p3, p2))
        u24 = np.subtract(p4, p2)/np.linalg.norm(np.subtract(p4, p2))
        u31 = -1 * u13 
        u32 = -1 * u23
        u34 = np.subtract(p4, p3)/np.linalg.norm(np.subtract(p4, p3))
        u41 = -1 * u14
        u42 = -1 * u24
        u43 = -1 * u34


        # cortical_flow = np.multiply(0.000527*time, np.e**(-0.01466569*time))
        cortical_flow_r = np.multiply(0.000345*time, np.e**(-0.012732*time))
        cortical_flow_l = np.multiply(0.00071*time, np.e**(-0.0166*time))

        # equation 1
        p1_prime = t_final * (self.B * ((np.linalg.norm(p1-p2) - 1) * u12 + 
                                        (np.linalg.norm(p1-p2) - 1) * u13 - 
                                        (p1z - 0.5) * k_hat) + 
                                self.A * cortical_flow_r * 
                                        (np.cross(u21, u24) - 
                                        np.cross(u12, u13) -
                                        np.cross(u13, k_hat)))
        # equation 2
        p2_prime = t_final * (self.B * ((np.linalg.norm(p2-p1) - 1) * u21 + 
                                        (np.linalg.norm(p2-p4) - 1) * u24 - 
                                        (p2z - 0.5) * k_hat) + 
                                self.A * cortical_flow_r * 
                                        (np.cross(u12, u13) -
                                        np.cross(u21, u24) -
                                        np.cross(u24, k_hat)))

        # equation 3
        p3_prime = t_final * (self.B * ((np.linalg.norm(p3-p1) - 1) * u31 + 
                                        (np.linalg.norm(p3-p4) - 1) * u34 - 
                                        (p3z - 0.5) * k_hat) + 
                                self.A * cortical_flow_l * 
                                        (np.cross(u43, u42) -
                                        np.cross(u34, u31) -
                                        np.cross(u31, k_hat)))

        # equation 4
        p4_prime = t_final * (self.B * ((np.linalg.norm(p4-p2) - 1) * u42 +
                                        (np.linalg.norm(p4-p3) - 1) * u43 - 
                                        (p4z - 0.5) * k_hat) + 
                                self.A * cortical_flow_l * 
                                        (np.cross(u34, u31) -
                                        np.cross(u43, u42) -
                                        np.cross(u42, k_hat)))
        
        # applies spring force across cells in next iteration
        if self.system.info.dist_14 <= 1:
                p1_prime += t_final * self.B * (np.linalg.norm(p1-p4) - 1) * u14
                p4_prime += t_final * self.B * (np.linalg.norm(p4-p1) - 1) * u41

        if self.system.info.dist_23 <= 1:
                p2_prime += t_final * self.B * (np.linalg.norm(p2-p3) - 1) * u23
                p3_prime += t_final * self.B * (np.linalg.norm(p3-p2) - 1) * u32

        return ModelAB(
              self.A,
              self.B,
              FourCellSystem(Cell(p1 + config.h * p1_prime),
                              Cell(p2 + config.h * p2_prime),
                              Cell(p3 + config.h * p3_prime),
                              Cell(p4 + config.h * p4_prime)))
        