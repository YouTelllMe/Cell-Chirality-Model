import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation


def euler(func, start_vec, h, tau, A, B, t_final):
    """
    Perform euler's method by repeatedly calling the given function. Note that this function
    is programmed to handle systems of 12 equations (4 position vectors in succession)
    """

    euler_coords = []
    distance = {"12": [1], "13": [1], "14": [1 * np.sqrt(2)],
                "23": [1 * np.sqrt(2)], "24": [1], "34": [1]}
    angle = {"dorsal1": [90], "dorsal2": [90], "anterior1": [0],
                "anterior2": [0]}
    curr_pos = start_vec
    euler_coords.append(curr_pos)

    # take "steps" steps
    for time in tau:
        curr_velocity = func(curr_pos, A, B, time, t_final)
        curr_pos = curr_pos + h * curr_velocity
        euler_coords.append(curr_pos)

        #positions
        position_1 = np.array([curr_pos[0], curr_pos[1], curr_pos[2]])
        position_2 = np.array([curr_pos[3], curr_pos[4], curr_pos[5]])
        position_3 = np.array([curr_pos[6], curr_pos[7], curr_pos[8]])
        position_4 = np.array([curr_pos[9], curr_pos[10], curr_pos[11]])

        axis_1 = position_1-position_3
        axis_2 = position_2-position_4

        dorsal_ax_1 = np.array([axis_1[0], axis_1[1]])
        dorsal_ax_2 = np.array([axis_2[0], axis_2[1]])
        anterior_ax_1 = np.array([axis_1[1], axis_1[2]])
        anterior_ax_2 = np.array([axis_2[1], axis_2[2]])

        dorsal_0 = np.array([1,0])
        anterior_0 = np.array([1,0])

        dorsal_ang_1 = np.arccos(np.dot(dorsal_ax_1, dorsal_0)/(np.linalg.norm(dorsal_ax_1)))
        dorsal_ang_2 = np.arccos(np.dot(dorsal_ax_2, dorsal_0)/(np.linalg.norm(dorsal_ax_2)))
        anterior_ang_1 = np.arccos(np.dot(anterior_ax_1, anterior_0)/(np.linalg.norm(anterior_ax_1)))
        anterior_ang_2 = np.arccos(np.dot(anterior_ax_2, anterior_0)/(np.linalg.norm(anterior_ax_2)))

        if dorsal_ax_1[1] < 0:
            dorsal_ang_1 *= -1
        if dorsal_ax_2[1] < 0:
            dorsal_ang_2 *= -1
        if anterior_ax_1[1] < 0:
            anterior_ang_1 *= -1
        if anterior_ax_2[1] < 0:
            anterior_ang_2 *= -1

        angle["dorsal1"].append(dorsal_ang_1 * (180 / np.pi))
        angle["dorsal2"].append(dorsal_ang_2 * (180 / np.pi))
        angle["anterior1"].append(anterior_ang_1 * (180 / np.pi))
        angle["anterior2"].append(anterior_ang_2 * (180 / np.pi))

        #add to distances
        distance["12"].append(np.linalg.norm(np.subtract(position_1, position_2)))
        distance["13"].append(np.linalg.norm(np.subtract(position_1, position_3)))
        distance["14"].append(np.linalg.norm(np.subtract(position_1, position_4)))
        distance["23"].append(np.linalg.norm(np.subtract(position_2, position_3)))
        distance["24"].append(np.linalg.norm(np.subtract(position_2, position_4)))
        distance["34"].append(np.linalg.norm(np.subtract(position_3, position_4)))

    # rudimentary check 
    assert len(start_vec) % 3 == 0, "number of columns not divisible by 3"
    num_columns = len(start_vec)

    # data frames with index as column names for all vector components
    column_names=[str(col) for col in range(num_columns)]
    euler_df = pd.DataFrame(data=euler_coords, columns=column_names)
    distance_df = pd.DataFrame(distance)
    angle_df = pd.DataFrame(angle)

    # separate df into 4 vector dfs 
    output_eulers = []
    for col in range(num_columns):
        if col % 3 == 0:
            subdf_index = (col // 3) + 1
            euler_subdf = euler_df[[str(col), str(col+1), str(col+2)]].copy()
            euler_subdf.rename(columns={str(col): "x", str(col+1): "y", str(col+2): "z"}, inplace=True)
            output_eulers.append(euler_subdf)
            euler_subdf.to_csv(f"data_{subdf_index}.csv")

    distance_df.to_csv("distance.csv")
    angle_df.to_csv("fit_angle.csv")

    return [output_eulers, distance_df, angle_df]

#-----------------------------------------------------------------
"ANIMATION"

COLORS = ["blue", "orange", "green", "red"]

def animate(euler_df):
    """
    Takes a euler df and animates it in matplotlib, assumes 4 components
    """
    steps = len(euler_df[0].index)


    curves = []
    for curve_index in range(len(euler_df)):
        color = COLORS[curve_index % 4]
        new_curve, = AX.plot([],[],[],".", alpha=0.4, markersize=3, label=curve_index+1, c=color)
        curves.append(new_curve)

    # axes of rotation initialization
    axis1, = AX.plot([],[],[],":", alpha=0.4, markersize=0, linewidth=1, c="black")
    axis2, = AX.plot([],[],[],":", alpha=0.4, markersize=0, linewidth=1, c="black")

    # wall vectors initialization
    x1, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    x2, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    y1, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    y2, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    z1, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    z2, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")

    # animate, run update_replace at every step
    anim = FuncAnimation(FIG, update_replace, steps, 
                        fargs=(curves, axis1, axis2, 
                               x1, x2, y1, y2, z1, z2,
                               euler_df)
                        ,interval=1
                        ,repeat=True)

    # save animation from different angles
    AX.legend()
    anim.save("XYZ.gif")

    AX.view_init(90, -90, 0)
    anim.save("XY.gif")

    AX.view_init(0, -90, 0)
    anim.save("XZ.gif")

    AX.view_init(0, 0, 0)
    anim.save("YZ.gif")


def update_replace(frame, curves, axis1, axis2,
                   x1, x2, y1, y2, z1, z2, data):

    # update position vector
    for curve_index in range(len(curves)):
        x = data[curve_index]["x"][frame]
        y = data[curve_index]["y"][frame]
        z = data[curve_index]["z"][frame]

        ball = generate_ball([x,y,z], 0.5)
        ball["x"].append(x)
        ball["y"].append(y)
        ball["z"].append(z)

        curves[curve_index].set_data([ball["x"], ball["y"]])
        curves[curve_index].set_3d_properties(ball["z"])

    # update wall cell axis
    x1.set_data([[-SIZE, -SIZE],[data[0]["y"][frame], data[2]["y"][frame]]])
    x1.set_3d_properties([data[0]["z"][frame], data[2]["z"][frame]])
    x2.set_data([[-SIZE, -SIZE],[data[1]["y"][frame], data[3]["y"][frame]]])
    x2.set_3d_properties([data[1]["z"][frame], data[3]["z"][frame]])

    y1.set_data([[data[0]["x"][frame], data[2]["x"][frame]],[SIZE, SIZE]])
    y1.set_3d_properties([data[0]["z"][frame], data[2]["z"][frame]])
    y2.set_data([[data[1]["x"][frame], data[3]["x"][frame]],[SIZE, SIZE]])
    y2.set_3d_properties([data[1]["z"][frame], data[3]["z"][frame]])

    z1.set_data([[data[0]["x"][frame], data[2]["x"][frame]],[data[0]["y"][frame], data[2]["y"][frame]]])
    z1.set_3d_properties([-SIZE, -SIZE])
    z2.set_data([[data[1]["x"][frame], data[3]["x"][frame]],[data[1]["y"][frame], data[3]["y"][frame]]])
    z2.set_3d_properties([-SIZE, -SIZE])


    # update axes of rotation
    axis1.set_data([[data[0]["x"][frame], data[2]["x"][frame]],[data[0]["y"][frame], data[2]["y"][frame]]])
    axis1.set_3d_properties([data[0]["z"][frame], data[2]["z"][frame]])
    axis2.set_data([[data[1]["x"][frame], data[3]["x"][frame]],[data[1]["y"][frame], data[3]["y"][frame]]])
    axis2.set_3d_properties([data[1]["z"][frame], data[3]["z"][frame]])

    

def generate_ball(position, radius):
    delta = np.pi / 20
    rotation_matrix = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])

    center_x = position[0]
    center_y = position[1]
    center_z = position[2]

    curr_x = 0
    curr_y = 0
    curr_z = radius
    ref_z = 0

    x = [center_x+curr_x]
    y = [center_y+curr_y]
    z = [center_z+curr_z]


    for phi in range(20):
        phi_vector = np.matmul(rotation_matrix, [ref_z, curr_z])
        ref_z = phi_vector[0]
        curr_z = phi_vector[1]
        curr_x = ref_z
        curr_y = 0
        for theta in range(40):
            theta_vector = np.matmul(rotation_matrix, [curr_x, curr_y])
            curr_x = theta_vector[0]
            curr_y = theta_vector[1]
            x.append(center_x+curr_x)
            y.append(center_y+curr_y)
            z.append(center_z+curr_z)
    
    return {"x": x, "y": y, "z": z}

#-----------------------------------------------------------------
"FOR TESTING"
def sample_func(vector):
    """Sample function, Euler for x^2. (x', y', z') = (1, 2x, 0)"""
    return np.array([1,2 * vector[0],0])

#----------------------------------------------------------------
"SCRIPT"


# non dimensionalized parameter

def physics_system(vector, A, B, time, t_final):
    """
    4 cell model physics system, used as "func" for Euler
    """
    
    # position vectors
    p1 = np.array([vector[0], vector[1], vector[2]])
    p2 = np.array([vector[3], vector[4], vector[5]])
    p3 =  np.array([vector[6], vector[7], vector[8]])
    p4 =  np.array([vector[9], vector[10], vector[11]])

    p1z = vector[2]
    p2z = vector[5]
    p3z = vector[8]
    p4z = vector[11]

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

    # equation 1
    p1_prime = t_final * (B * ((np.linalg.norm(np.subtract(p1, p2)) - 1) * u12 
                + (np.linalg.norm(np.subtract(p2, p4)) - 1) * u13
                + (np.linalg.norm(np.subtract(p1, p4)) - np.sqrt(2)) * u14
                - (p1z - 1 / 2) * k_hat)
                + A * np.multiply(0.000527 * time,np.e**(-0.01466569 * time)) * 
                (np.cross(u21, u24)-np.cross(u12, u13)-np.cross(u13, k_hat)))
    # p1_prime_normalized = p1_prime / np.linalg.norm(p1_prime)

    # equation 2
    p2_prime = t_final * (B * ((np.linalg.norm(np.subtract(p2, p1)) - 1) * u21
                + (np.linalg.norm(np.subtract(p2, p4)) - 1) * u24
                + (np.linalg.norm(np.subtract(p2, p3)) - np.sqrt(2)) * u23
                - (p2z - 1 / 2) * k_hat)
                + A * np.multiply(0.000527 * time,np.e**(-0.01466569 * time)) * 
                (np.cross(u12, u13)-np.cross(u21, u24)-np.cross(u24, k_hat)))
    # p2_prime_normalized = p2_prime / np.linalg.norm(p2_prime)

    # equation 3
    p3_prime = t_final * (B * ((np.linalg.norm(np.subtract(p3, p1)) - 1) * u31
                + (np.linalg.norm(np.subtract(p3, p4)) - 1) * u34
                + (np.linalg.norm(np.subtract(p3, p2)) - np.sqrt(2)) * u32
                - (p3z - 1 / 2) * k_hat)
                + A * np.multiply(0.000527 * time,np.e**(-0.01466569 * time)) * 
                (np.cross(u43, u42)-np.cross(u34, u31)-np.cross(u31, k_hat)))
    # p3_prime_normalized = p3_prime / np.linalg.norm(p3_prime)

    # equation 4
    p4_prime = t_final * (B * ((np.linalg.norm(np.subtract(p4, p2)) - 1) * u42 
                + (np.linalg.norm(np.subtract(p4, p3)) - 1) * u43
                + (np.linalg.norm(np.subtract(p4, p1)) - np.sqrt(2)) * u41
                - (p4z - 1 / 2) * k_hat)
                + A * np.multiply(0.000527 * time,np.e**(-0.01466569 * time)) * 
                (np.cross(u34, u31)-np.cross(u43, u42)-np.cross(u42, k_hat)))
    # p4_prime_normalized = p4_prime / np.linalg.norm(p4_prime)

    
    return np.concatenate((p1_prime, p2_prime, p3_prime, p4_prime), axis=None)



