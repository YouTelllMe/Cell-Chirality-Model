import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation


def euler(func, start_vec, h, steps):
    euler_coords = [start_vec]
    curr_pos = start_vec
    curr_velocity = func(start_vec)

    for _ in range(steps):
        curr_pos = curr_pos + h * curr_velocity
        curr_velocity = func(curr_pos)
        euler_coords.append(curr_pos)
    
    assert len(start_vec) % 3 == 0, "number of columns not divisible by 3"
    num_columns = len(start_vec)
    euler_df = pd.DataFrame(data=euler_coords, columns=[str(col) for col in range(num_columns)])

    output_eulers = []
    for col in range(num_columns):
        if col % 3 == 0:
            subdf_index = (col // 3) + 1
            euler_subdf = euler_df[[str(col), str(col+1), str(col+2)]].copy()
            euler_subdf.rename(columns={str(col): "x", str(col+1): "y", str(col+2): "z"}, inplace=True)
            output_eulers.append(euler_subdf)
            euler_subdf.to_csv(f"data_{subdf_index}.csv")

    return output_eulers


def plot_euler(euler_df):
    AX.plot(euler_df["x"], euler_df["y"], euler_df["z"])

#-----------------------------------------------------------------
"ANIMATION"

COLORS = ["blue", "orange", "green", "red"]

def animate(euler_df):
    steps = len(euler_df[0].index)

    curves = []
    curve_supports = []
    for curve_index in range(len(euler_df)):
        color = COLORS[curve_index % 4]
        new_curve, = AX.plot([],[],[],".", alpha=0.4, markersize=20, label=curve_index+1, c=color)
        x_support, = AX.plot([],[],[],"o", alpha=0.4, markersize=3, linewidth=1, c=color)
        y_support, = AX.plot([],[],[],"o", alpha=0.4, markersize=3, linewidth=1, c=color)
        z_support, = AX.plot([],[],[],"o", alpha=0.4, markersize=3, linewidth=1, c=color)

        curves.append(new_curve)
        curve_supports.append([x_support, y_support, z_support])

    # only for 4 cells
    axis1, = AX.plot([],[],[],":", alpha=0.4, markersize=0, c="black")
    axis2, = AX.plot([],[],[],":", alpha=0.4, markersize=0, c="black")




    anim = FuncAnimation(FIG, update_replace, steps, 
                        fargs=(curves, curve_supports, axis1, axis2, euler_df)
                        ,interval=500
                        ,repeat=True)


    AX.legend()
    anim.save("XYZ.gif")

    AX.view_init(90, -90, 0)
    anim.save("XY.gif")

    AX.view_init(0, -90, 0)
    anim.save("XZ.gif")

    AX.view_init(0, 0, 0)
    anim.save("YZ.gif")


# def update(frame, curve, x, y, z, data):
#     x.append(data["x"][frame])
#     y.append(data["y"][frame])
#     z.append(data["z"][frame])
#     curve.set_data(x, y)
#     curve.set_3d_properties(z)
#     return curve, 

def update_replace(frame, curves, curve_supports, axis1, axis2, data):

    for curve_index in range(len(curves)):
        x = data[curve_index]["x"][frame]
        y = data[curve_index]["y"][frame]
        z = data[curve_index]["z"][frame]

        curves[curve_index].set_data([[x], [y]])
        curves[curve_index].set_3d_properties([z])

        # x supports
        # curve_supports[curve_index][0].set_data([[x, x], [0, y]])
        # curve_supports[curve_index][0].set_3d_properties([0, z])

        # # y supports
        # curve_supports[curve_index][1].set_data([[0, x], [y, y]])
        # curve_supports[curve_index][1].set_3d_properties([0, z])

        # # z supports
        # curve_supports[curve_index][2].set_data([[0, x], [0, y]])
        # curve_supports[curve_index][2].set_3d_properties([z, z])

        # xy supports
        curve_supports[curve_index][0].set_data([[x, x], [y, y]])
        curve_supports[curve_index][0].set_3d_properties([z, -SIZE])

        # xz supports
        curve_supports[curve_index][1].set_data([[x, -SIZE], [y, y]])
        curve_supports[curve_index][1].set_3d_properties([z, z])

        # yz supports
        curve_supports[curve_index][2].set_data([[x, x], [y, SIZE]])
        curve_supports[curve_index][2].set_3d_properties([z, z])



    # only for 4 cells
    axis1.set_data([[data[0]["x"][frame], data[2]["x"][frame]],[data[0]["y"][frame], data[2]["y"][frame]]])
    axis1.set_3d_properties([data[0]["z"][frame], data[2]["z"][frame]])
    axis2.set_data([[data[1]["x"][frame], data[3]["x"][frame]],[data[1]["y"][frame], data[3]["y"][frame]]])
    axis2.set_3d_properties([data[1]["z"][frame], data[3]["z"][frame]])

    

#-----------------------------------------------------------------
"FOR TESTING"
def sample_func(vector):
    # sample function, Euler for x^2. (x', y', z') = (1, 2x, 0)
    return np.array([1,2 * vector[0],0])

#----------------------------------------------------------------
"SCRIPT"

OMEGA = 1
L = 2
K = 1
D = 1
C = 1

def physics_system(vector):
    
    # position vectors
    p1 = np.array([vector[0], vector[1], vector[2]])
    p2 = np.array([vector[3], vector[4], vector[5]])
    p3 =  np.array([vector[6], vector[7], vector[8]])
    p4 =  np.array([vector[9], vector[10], vector[11]])

    p1z = vector[2]
    p2z = vector[5]
    p3z =  vector[8]
    p4z =  vector[11]

    k_hat = np.array([0,0,1])

    # unit vectors 
    u13 = np.subtract(p3, p1)/np.linalg.norm(np.subtract(p3, p1))
    u12 = np.subtract(p2, p1)/np.linalg.norm(np.subtract(p2, p1))
    u24 = np.subtract(p4, p2)/np.linalg.norm(np.subtract(p4, p2))
    u21 = -1 * u12
    u31 = -1 * u13
    u34 = np.subtract(p4, p3)/np.linalg.norm(np.subtract(p4, p3))
    u42 = -1 * u24
    u43 = -1 * u34

    # equation 1
    p1_prime = (K * (np.linalg.norm(np.subtract(p1, p2)) - L) * u12 
                + K * (np.linalg.norm(np.subtract(p2, p4)) - L) * u13
                - K * (p1z - L / 2) * k_hat
                + D * L / 2 * OMEGA * (np.cross(u21, u24)-np.cross(u12, u13)-np.cross(u13, k_hat))) / C
    p1_prime_normalized = p1_prime / np.linalg.norm(p1_prime)

    # equation 2
    p2_prime = (K * (np.linalg.norm(np.subtract(p2, p1)) - L) * u21
                + K * (np.linalg.norm(np.subtract(p2, p4)) - L) * u24
                - K * (p2z - L / 2) * k_hat
                + D * L / 2 * OMEGA * (np.cross(u12, u13)-np.cross(u21, u24)-np.cross(u24, k_hat))) / C
    p2_prime_normalized = p2_prime / np.linalg.norm(p2_prime)

    # equation 3
    p3_prime = (K * (np.linalg.norm(np.subtract(p3, p1)) - L) * u31
                + K * (np.linalg.norm(np.subtract(p3, p4)) - L) * u34
                - K * (p3z - L / 2) * k_hat
                + D * L / 2 * OMEGA * (np.cross(u43, u42)-np.cross(u34, u31)-np.cross(u31, k_hat))) / C
    p3_prime_normalized = p3_prime / np.linalg.norm(p3_prime)

    # equation 4
    p4_prime = (K * (np.linalg.norm(np.subtract(p4, p2)) - L) * u42 
                + K * (np.linalg.norm(np.subtract(p4, p3)) - L) * u43
                - K * (p4z - L / 2) * k_hat
                + D * L / 2 * OMEGA * (np.cross(u34, u31)-np.cross(u43, u42)-np.cross(u42, k_hat))) / C
    p4_prime_normalized = p4_prime / np.linalg.norm(p4_prime)

    
    return np.concatenate((p1_prime_normalized, p2_prime_normalized, p3_prime_normalized, p4_prime_normalized), axis=None)





# setup
FIG = plt.figure()
AX = FIG.add_subplot(projection="3d")
SIZE = 2
# Setting the axes properties
AX.set(xlim3d=(-SIZE, SIZE), xlabel='X')
AX.set(ylim3d=(-SIZE, SIZE), ylabel='Y')
AX.set(zlim3d=(-SIZE, SIZE), zlabel='Z')

# AX.grid(False)

# xaxis, = AX.plot([SIZE, -SIZE],[0, 0],[0, 0],"-", alpha=0.2, c="black")
# yaxis, = AX.plot([0, 0],[SIZE, -SIZE],[0, 0],"-", alpha=0.2, c="black")
# zaxis, = AX.plot([0, 0],[0, 0],[SIZE, -SIZE],"-", alpha=0.2, c="black")

# print(physics_system([-1,1,1,1,1,1,-1,-1,1,-1,1,1]))


euler_df = euler(physics_system, np.array([-1,1,1, 1,1,1, -1,-1,1, 1,-1,1]), 0.05, 20)
animate(euler_df)
plt.show()



