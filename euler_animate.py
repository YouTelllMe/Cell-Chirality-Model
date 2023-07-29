import numpy as np
from matplotlib.animation import FuncAnimation


COLORS = ["blue", "orange", "green", "red"]

def animate(FIG, AX, euler_df, SIZE):
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
                               euler_df, SIZE)
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
                   x1, x2, y1, y2, z1, z2, data, SIZE):

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