import numpy as np
import config


def model_ABC(curr_pos, A, B, time, diag14: bool, diag23: bool, C):
      """
      4 cell model physics system, used as "func" for Euler, p2-cell fitted
      """
    
      t_final = config.T_FINAL

      # position vectors
      p1 = np.array([curr_pos[0], curr_pos[1], curr_pos[2]])
      p2 = np.array([curr_pos[3], curr_pos[4], curr_pos[5]])
      p3 =  np.array([curr_pos[6], curr_pos[7], curr_pos[8]])
      p4 =  np.array([curr_pos[9], curr_pos[10], curr_pos[11]])
      p2_cell = np.array([0.5+np.cos(C)*0.5*np.sqrt(3),0,0.5-np.sin(C)*0.5*np.sqrt(3)])


      p1z = curr_pos[2]
      p2z = curr_pos[5]
      p3z = curr_pos[8]
      p4z = curr_pos[11]
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

      u22_cell = np.subtract(p2_cell, p2)/np.linalg.norm(np.subtract(p2_cell, p2))
      u42_cell = np.subtract(p2_cell, p4)/np.linalg.norm(np.subtract(p2_cell, p4))


      # cortical_flow = np.multiply(0.000527*time, np.e**(-0.01466569*time))
      cortical_flow_r = np.multiply(0.000345*time, np.e**(-0.012732*time))
      cortical_flow_l = np.multiply(0.00071*time, np.e**(-0.0166*time))

      # equation 1
      p1_prime = t_final * (B * ((np.linalg.norm(p1-p2) - 1) * u12 + 
                                    (np.linalg.norm(p2-p4) - 1) * u13 - 
                                    (p1z - 0.5) * k_hat) + 
                              A * cortical_flow_r * 
                                    (np.cross(u21, u24) - 
                                    np.cross(u12, u13) -
                                    np.cross(u13, k_hat)))

      # equation 2
      p2_prime = t_final * (B * ((np.linalg.norm(p2-p1) - 1) * u21 + 
                                    (np.linalg.norm(p2-p4) - 1) * u24 - 
                                    (p2z - 0.5) * k_hat + 
                                    (np.linalg.norm(p2-p2_cell) - 1) * u22_cell) + 
                              A * cortical_flow_r * 
                                    (np.cross(u12, u13) -
                                    np.cross(u21, u24) -
                                    np.cross(u24, k_hat) + 
                                    np.cross(u24, u22_cell)))

      # equation 3
      p3_prime = t_final * (B * ((np.linalg.norm(p3-p1) - 1) * u31 + 
                                    (np.linalg.norm(p3-p4) - 1) * u34 - 
                                    (p3z - 0.5) * k_hat) + 
                              A * cortical_flow_l * 
                                    (np.cross(u43, u42) -
                                    np.cross(u34, u31) -
                                    np.cross(u31, k_hat)))

      # equation 4
      p4_prime = t_final * (B * ((np.linalg.norm(p4-p2) - 1) * u42 +
                                    (np.linalg.norm(p4-p3) - 1) * u43 - 
                                    (p4z - 0.5) * k_hat + 
                                    (np.linalg.norm(p4-p2_cell) - 1) * u42_cell) + 
                              A * cortical_flow_l * 
                                    (np.cross(u34, u31) -
                                    np.cross(u43, u42) -
                                    np.cross(u42, k_hat) -
                                    np.cross(u42, u42_cell)))
      
      if diag14:
            p1_prime += t_final * B * (np.linalg.norm(p1-p4) - 1) * u14
            p4_prime += t_final * B * (np.linalg.norm(p4-p1) - 1) * u41

      if diag23:
            p2_prime += t_final * B * (np.linalg.norm(p2-p3) - 1) * u23
            p3_prime += t_final * B * (np.linalg.norm(p3-p2) - 1) * u32

      return np.concatenate((p1_prime, p2_prime, p3_prime, p4_prime), axis=None)


def model_AB(curr_pos, A, B, time, diag14: bool, diag23: bool):
      """
      4 cell model physics system, used as "func" for Euler, p2-cell behind
      """
    
      t_final = config.T_FINAL

      # position vectors
      p1 = np.array([curr_pos[0], curr_pos[1], curr_pos[2]])
      p2 = np.array([curr_pos[3], curr_pos[4], curr_pos[5]])
      p3 =  np.array([curr_pos[6], curr_pos[7], curr_pos[8]])
      p4 =  np.array([curr_pos[9], curr_pos[10], curr_pos[11]])
      p2_cell = np.array([1.25308708772,0,0.07237886124])


      p1z = curr_pos[2]
      p2z = curr_pos[5]
      p3z = curr_pos[8]
      p4z = curr_pos[11]
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

      u22_cell = np.subtract(p2_cell, p2)/np.linalg.norm(np.subtract(p2_cell, p2))
      u42_cell = np.subtract(p2_cell, p4)/np.linalg.norm(np.subtract(p2_cell, p4))


      cortical_flow = np.multiply(0.000527*time, np.e**(-0.01466569*time))
      # cortical_flow_r = np.multiply(0.000345*time, np.e**(-0.012732*time))
      # cortical_flow_l = np.multiply(0.00071*time, np.e**(-0.0166*time))

      # equation 1
      p1_prime = t_final * (B * ((np.linalg.norm(p1-p2) - 1) * u12 + 
                                    (np.linalg.norm(p2-p4) - 1) * u13 - 
                                    (p1z - 0.5) * k_hat) + 
                              A * cortical_flow * 
                                    (np.cross(u21, u24) - 
                                    np.cross(u12, u13) -
                                    np.cross(u13, k_hat)))

      # equation 2
      p2_prime = t_final * (B * ((np.linalg.norm(p2-p1) - 1) * u21 + 
                                    (np.linalg.norm(p2-p4) - 1) * u24 - 
                                    (p2z - 0.5) * k_hat + 
                                    (np.linalg.norm(p2-p2_cell) - 1) * u22_cell) + 
                              A * cortical_flow * 
                                    (np.cross(u12, u13) -
                                    np.cross(u21, u24) -
                                    np.cross(u24, k_hat) + 
                                    np.cross(u24, u22_cell)))

      # equation 3
      p3_prime = t_final * (B * ((np.linalg.norm(p3-p1) - 1) * u31 + 
                                    (np.linalg.norm(p3-p4) - 1) * u34 - 
                                    (p3z - 0.5) * k_hat) + 
                              A * cortical_flow * 
                                    (np.cross(u43, u42) -
                                    np.cross(u34, u31) -
                                    np.cross(u31, k_hat)))

      # equation 4
      p4_prime = t_final * (B * ((np.linalg.norm(p4-p2) - 1) * u42 +
                                    (np.linalg.norm(p4-p3) - 1) * u43 - 
                                    (p4z - 0.5) * k_hat + 
                                    (np.linalg.norm(p4-p2_cell) - 1) * u42_cell) + 
                              A * cortical_flow * 
                                    (np.cross(u34, u31) -
                                    np.cross(u43, u42) -
                                    np.cross(u42, k_hat) -
                                    np.cross(u42, u42_cell)))
      
      if diag14:
            p1_prime += t_final * B * (np.linalg.norm(p1-p4) - 1) * u14
            p4_prime += t_final * B * (np.linalg.norm(p4-p1) - 1) * u41

      if diag23:
            p2_prime += t_final * B * (np.linalg.norm(p2-p3) - 1) * u23
            p3_prime += t_final * B * (np.linalg.norm(p3-p2) - 1) * u32

      return np.concatenate((p1_prime, p2_prime, p3_prime, p4_prime), axis=None)



# def model_AB(curr_pos, A, B, time, diag14: bool, diag23: bool):
#     """
#     4 cell model physics system, used as "func" for Euler, no p2-cell
#     """
    
#     t_final = config.T_FINAL

#     # position vectors
#     p1 = np.array([curr_pos[0], curr_pos[1], curr_pos[2]])
#     p2 = np.array([curr_pos[3], curr_pos[4], curr_pos[5]])
#     p3 =  np.array([curr_pos[6], curr_pos[7], curr_pos[8]])
#     p4 =  np.array([curr_pos[9], curr_pos[10], curr_pos[11]])
#     p2_cell = np.array([0.5+0.5*np.sqrt(3),0,0.5])


#     p1z = curr_pos[2]
#     p2z = curr_pos[5]
#     p3z = curr_pos[8]
#     p4z = curr_pos[11]
#     k_hat = np.array([0,0,1])

#     # unit vectors 
#     u12 = np.subtract(p2, p1)/np.linalg.norm(np.subtract(p2, p1))
#     u13 = np.subtract(p3, p1)/np.linalg.norm(np.subtract(p3, p1))
#     u14 = np.subtract(p4, p1)/np.linalg.norm(np.subtract(p4, p1))
#     u21 = -1 * u12
#     u23 = np.subtract(p3, p2)/np.linalg.norm(np.subtract(p3, p2))
#     u24 = np.subtract(p4, p2)/np.linalg.norm(np.subtract(p4, p2))
#     u31 = -1 * u13 
#     u32 = -1 * u23
#     u34 = np.subtract(p4, p3)/np.linalg.norm(np.subtract(p4, p3))
#     u41 = -1 * u14
#     u42 = -1 * u24
#     u43 = -1 * u34

#     cortical_flow = np.multiply(0.000527*time, np.e**(-0.01466569*time))
# #     cortical_flow_r = np.multiply(0.000345*time, np.e**(-0.012732*time))
# #     cortical_flow_l = np.multiply(0.00071*time, np.e**(-0.0166*time))

#     # equation 1
#     p1_prime = t_final * (B * ((np.linalg.norm(p1-p2) - 1) * u12 + 
#                                 (np.linalg.norm(p2-p4) - 1) * u13 - 
#                                 (p1z - 0.5) * k_hat) + 
#                           A * cortical_flow * 
#                                 (np.cross(u21, u24) - 
#                                  np.cross(u12, u13) -
#                                  np.cross(u13, k_hat)))

#     # equation 2
#     p2_prime = t_final * (B * ((np.linalg.norm(p2-p1) - 1) * u21 + 
#                                 (np.linalg.norm(p2-p4) - 1) * u24 - 
#                                 (p2z - 0.5) * k_hat) + 
#                           A * cortical_flow * 
#                                 (np.cross(u12, u13) -
#                                  np.cross(u21, u24) -
#                                  np.cross(u24, k_hat)))

#     # equation 3
#     p3_prime = t_final * (B * ((np.linalg.norm(p3-p1) - 1) * u31 + 
#                                 (np.linalg.norm(p3-p4) - 1) * u34 - 
#                                 (p3z - 0.5) * k_hat) + 
#                           A * cortical_flow * 
#                                 (np.cross(u43, u42) -
#                                  np.cross(u34, u31) -
#                                  np.cross(u31, k_hat)))

#     # equation 4
#     p4_prime = t_final * (B * ((np.linalg.norm(p4-p2) - 1) * u42 +
#                                 (np.linalg.norm(p4-p3) - 1) * u43 - 
#                                 (p4z - 0.5) * k_hat) + 
#                           A * cortical_flow * 
#                                 (np.cross(u34, u31) -
#                                  np.cross(u43, u42) -
#                                  np.cross(u42, k_hat)))
    
#     if diag14:
#         p1_prime += t_final * B * (np.linalg.norm(p1-p4) - 1) * u14
#         p4_prime += t_final * B * (np.linalg.norm(p4-p1) - 1) * u41

#     if diag23:
#         p2_prime += t_final * B * (np.linalg.norm(p2-p3) - 1) * u23
#         p3_prime += t_final * B * (np.linalg.norm(p3-p2) - 1) * u32

#     return np.concatenate((p1_prime, p2_prime, p3_prime, p4_prime), axis=None)

