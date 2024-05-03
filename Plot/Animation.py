import numpy as np

def generate_ball(self, position: tuple[float, float, float], radius: float):
    """
    Return evenly spaced coordinates on the sphere with center at position with radius radius.
    """

    delta = np.pi / 20
    rotation_matrix = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])

    center_x, center_y, center_z = position
    curr_x, curr_y, curr_z = 0, 0, radius
    ref_z = 0
    x = [center_x+curr_x]
    y = [center_y+curr_y]
    z = [center_z+curr_z]

    for _ in range(20):
        phi_vector = np.matmul(rotation_matrix, [ref_z, curr_z])
        ref_z = phi_vector[0]
        curr_z = phi_vector[1]
        curr_x = ref_z
        curr_y = 0
        for _ in range(40):
            theta_vector = np.matmul(rotation_matrix, [curr_x, curr_y])
            curr_x = theta_vector[0]
            curr_y = theta_vector[1]
            x.append(center_x+curr_x)
            y.append(center_y+curr_y)
            z.append(center_z+curr_z)
    
    return {"x": x, "y": y, "z": z}