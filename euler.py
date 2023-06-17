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
            euler_subdf = euler_df[[str(col), str(col+1), str(col+2)]].copy()
            euler_subdf.rename(columns={str(col): "x", str(col+1): "y", str(col+2): "z"}, inplace=True)
            output_eulers.append(euler_subdf)

    return output_eulers


def plot_euler(euler_df):
    AX.plot(euler_df["x"], euler_df["y"], euler_df["z"])

#-----------------------------------------------------------------
"ANIMATION"
def animate(euler_df):
    steps = len(euler_df[0].index)

    curve, = AX.plot([],[],[],".", alpha=0.3, markersize=30)
    anim = FuncAnimation(FIG, update_replace, steps, 
                        fargs=(curve, euler_df)
                        ,interval=500
                        ,repeat=True)
    plt.show()
    anim.save("movie.gif")

# def update(frame, curve, x, y, z, data):
#     x.append(data["x"][frame])
#     y.append(data["y"][frame])
#     z.append(data["z"][frame])
#     curve.set_data(x, y)
#     curve.set_3d_properties(z)
#     return curve, 

def update_replace(frame, curve, data):

    x, y, z = [], [], []

    for euler in data:
        x.append(euler["x"][frame])
        y.append(euler["y"][frame])
        z.append(euler["z"][frame])
    
    curve.set_data([x, y])
    curve.set_3d_properties([z])
    
    return curve, 

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
# Setting the axes properties
AX.set(xlim3d=(-2, 2), xlabel='X')
AX.set(ylim3d=(-2, 2), ylabel='Y')
AX.set(zlim3d=(-2, 2), zlabel='Z')

# print(physics_system([-1,1,1,1,1,1,-1,-1,1,-1,1,1]))


euler_df = euler(physics_system, np.array([-1,1,1, 1,1,1, -1,-1,1, 1,-1,1]), 0.05, 20)
animate(euler_df)
plt.show()



