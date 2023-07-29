import numpy as np


def model_AB(curr_pos, A, B, time, t_final, diag14: bool, diag23: bool):
    """
    4 cell model physics system, used as "func" for Euler
    """
    
    # position vectors
    p1 = np.array([curr_pos[0], curr_pos[1], curr_pos[2]])
    p2 = np.array([curr_pos[3], curr_pos[4], curr_pos[5]])
    p3 =  np.array([curr_pos[6], curr_pos[7], curr_pos[8]])
    p4 =  np.array([curr_pos[9], curr_pos[10], curr_pos[11]])

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


    cortical_flow = np.multiply(0.000527*time, np.e**(-0.01466569*time))

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
                                (p2z - 0.5) * k_hat) + 
                          A * cortical_flow * 
                                (np.cross(u12, u13) -
                                 np.cross(u21, u24) -
                                 np.cross(u24, k_hat)))

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
                                (p4z - 0.5) * k_hat) + 
                          A * cortical_flow * 
                                (np.cross(u34, u31) -
                                 np.cross(u43, u42) -
                                 np.cross(u42, k_hat)))
    
    if diag14:
        p1_prime += t_final * B * (np.linalg.norm(p1-p4) - 1) * u14
        p4_prime += t_final * B * (np.linalg.norm(p4-p1) - 1) * u41

    if diag23:
        p2_prime += t_final * B * (np.linalg.norm(p2-p3) - 1) * u23
        p3_prime += t_final * B * (np.linalg.norm(p3-p2) - 1) * u32

    return np.concatenate((p1_prime, p2_prime, p3_prime, p4_prime), axis=None)


#-----------------------------------------------------------------------------------------------------#

# falty; retired 
# def model_Alt(vector, A, l, ck, t):
#     """
#     4 cell model physics system, used as "func" for Euler
#     """
    
#     # position vectors
#     p1 = np.array([vector[0], vector[1], vector[2]])
#     p2 = np.array([vector[3], vector[4], vector[5]])
#     p3 =  np.array([vector[6], vector[7], vector[8]])
#     p4 =  np.array([vector[9], vector[10], vector[11]])

#     p1z = vector[2]
#     p2z = vector[5]
#     p3z = vector[8]
#     p4z = vector[11]

#     k_hat = np.array([0,0,1])

#     # unit vectors 
#     u13 = np.subtract(p3, p1)/np.linalg.norm(np.subtract(p3, p1))
#     u12 = np.subtract(p2, p1)/np.linalg.norm(np.subtract(p2, p1))
#     u24 = np.subtract(p4, p2)/np.linalg.norm(np.subtract(p4, p2))
#     u21 = -1 * u12
#     u31 = -1 * u13 
#     u34 = np.subtract(p4, p3)/np.linalg.norm(np.subtract(p4, p3))
#     u42 = -1 * u24
#     u43 = -1 * u34

#     # equation 1
#     p1_prime = l * ((np.linalg.norm(np.subtract(p1, p2)) - 1) * u12 
#                 + (np.linalg.norm(np.subtract(p2, p4)) - 1) * u13
#                 - (p1z - 1 / 2) * k_hat
#                 + A * np.multiply((ck*t)**2,np.e**(-0.02169 * (ck*t))) * 
#                 (np.cross(u21, u24)-np.cross(u12, u13)-np.cross(u13, k_hat)))

#     # equation 2
#     p2_prime = l * ((np.linalg.norm(np.subtract(p2, p1)) - 1) * u21
#                 + (np.linalg.norm(np.subtract(p2, p4)) - 1) * u24
#                 - (p2z - 1 / 2) * k_hat
#                 + A * np.multiply((ck*t)**2,np.e**(-0.02169 * (ck*t))) * 
#                 (np.cross(u12, u13)-np.cross(u21, u24)-np.cross(u24, k_hat)))

#     # equation 3
#     p3_prime = l * ((np.linalg.norm(np.subtract(p3, p1)) - 1) * u31
#                 + (np.linalg.norm(np.subtract(p3, p4)) - 1) * u34
#                 - (p3z - 1 / 2) * k_hat
#                 + A * np.multiply((ck*t)**2,np.e**(-0.02169 * (ck*t))) * 
#                 (np.cross(u43, u42)-np.cross(u34, u31)-np.cross(u31, k_hat)))
    
#     # equation 4
#     p4_prime = l * ((np.linalg.norm(np.subtract(p4, p2)) - 1) * u42 
#                 + (np.linalg.norm(np.subtract(p4, p3)) - 1) * u43
#                 - (p4z - 1 / 2) * k_hat
#                 + A * np.multiply((ck*t)**2,np.e**(-0.02169 * (ck*t))) * 
#                 (np.cross(u34, u31)-np.cross(u43, u42)-np.cross(u42, k_hat)))
    
#     return np.concatenate((p1_prime, p2_prime, p3_prime, p4_prime), axis=None)
