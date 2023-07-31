import numpy as np
import matplotlib.pyplot as plt
import config
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pandas import DataFrame
from typing import Sequence


def animate(euler_df: Sequence[DataFrame]):
    """
    Takes a euler df and animates it in matplotlib, assumes 4 components
    """
    # setup figure and axes
    FIG = plt.figure()
    AX = FIG.add_subplot(projection="3d")

    # setting range 
    SIZE = 1
    AX.set(xlim3d=(-SIZE, SIZE), xlabel='X')
    AX.set(ylim3d=(-SIZE, SIZE), ylabel='Y')
    AX.set(zlim3d=(-SIZE, SIZE), zlabel='Z')

    # initialize 4 position vectors 
    positions_vectors = []
    animated_indicies = range(0, config.MODEL_STEPS, config.STEP_SCALE)
    for position_index in range(len(euler_df)):
        color = config.COLORS[position_index]
        new_position, = AX.plot([],[],[],".", alpha=0.4, markersize=3, label=position_index+1, c=color)
        positions_vectors.append(new_position)
        euler_df[position_index] = euler_df[position_index].iloc[animated_indicies].reset_index(drop=True)
    steps = len(euler_df[0].index)
    AX.legend()

    # initialize axes of rotation
    rotation_axis_1, = AX.plot([],[],[],":", alpha=0.4, markersize=0, linewidth=1, c="black")
    rotation_axis_2, = AX.plot([],[],[],":", alpha=0.4, markersize=0, linewidth=1, c="black")
    rotation_axes = (rotation_axis_1,
                     rotation_axis_2,)

    # initialize wall vectors 
    wall_x1, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    wall_x2, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    wall_y1, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    wall_y2, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    wall_z1, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    wall_z2, = AX.plot([],[],[],"-o", alpha=0.4, markersize=3, linewidth=1, c="black")
    wall_vectors = (wall_x1,
                    wall_x2,
                    wall_y1,
                    wall_y2,
                    wall_z1,
                    wall_z2,)

    # animate, run update_replace at every step
    anim = FuncAnimation(FIG, 
                         update_replace, 
                         steps, 
                         fargs=(euler_df,
                               SIZE,
                               positions_vectors, 
                               rotation_axes, 
                               wall_vectors, 
                               ),
                         interval=1,
                         repeat=True,
                         )

    # save animation from different angles
    anim.save(config.PLOT_XYZ)
    AX.view_init(90, -90, 0)
    anim.save(config.PLOT_XY)
    AX.view_init(0, -90, 0)
    anim.save(config.PLOT_XZ)
    AX.view_init(0, 0, 0)
    anim.save(config.PLOT_YZ)


def update_replace(frame, 
                   data, 
                   SIZE, 
                   positions, 
                   rotation_axes,
                   wall_vectors,):

    # update position vector, draw spheres
    for curve_index in range(len(positions)):
        x = data[curve_index]["x"][frame]
        y = data[curve_index]["y"][frame]
        z = data[curve_index]["z"][frame]
        ball = generate_ball([x,y,z], 0.5)
        ball["x"].append(x)
        ball["y"].append(y)
        ball["z"].append(z)
        positions[curve_index].set_data([ball["x"], ball["y"]])
        positions[curve_index].set_3d_properties(ball["z"])

    # update wall cell axis
    wall_vectors[0].set_data([[-SIZE, -SIZE],[data[0]["y"][frame], data[2]["y"][frame]]])
    wall_vectors[0].set_3d_properties([data[0]["z"][frame], data[2]["z"][frame]])
    wall_vectors[1].set_data([[-SIZE, -SIZE],[data[1]["y"][frame], data[3]["y"][frame]]])
    wall_vectors[1].set_3d_properties([data[1]["z"][frame], data[3]["z"][frame]])
    wall_vectors[2].set_data([[data[0]["x"][frame], data[2]["x"][frame]],[SIZE, SIZE]])
    wall_vectors[2].set_3d_properties([data[0]["z"][frame], data[2]["z"][frame]])
    wall_vectors[3].set_data([[data[1]["x"][frame], data[3]["x"][frame]],[SIZE, SIZE]])
    wall_vectors[3].set_3d_properties([data[1]["z"][frame], data[3]["z"][frame]])
    wall_vectors[4].set_data([[data[0]["x"][frame], data[2]["x"][frame]],[data[0]["y"][frame], data[2]["y"][frame]]])
    wall_vectors[4].set_3d_properties([-SIZE, -SIZE])
    wall_vectors[5].set_data([[data[1]["x"][frame], data[3]["x"][frame]],[data[1]["y"][frame], data[3]["y"][frame]]])
    wall_vectors[5].set_3d_properties([-SIZE, -SIZE])

    # update axes of rotation
    rotation_axes[0].set_data([[data[0]["x"][frame], data[2]["x"][frame]],[data[0]["y"][frame], data[2]["y"][frame]]])
    rotation_axes[0].set_3d_properties([data[0]["z"][frame], data[2]["z"][frame]])
    rotation_axes[1].set_data([[data[1]["x"][frame], data[3]["x"][frame]],[data[1]["y"][frame], data[3]["y"][frame]]])
    rotation_axes[1].set_3d_properties([data[1]["z"][frame], data[3]["z"][frame]])

    
def generate_ball(position: tuple[float, float, float], radius: float):
    """
    Return evenly spaced coordinates on the sphere with center at position with radius radius.
    """
    
    delta = np.pi / 20
    rotation_matrix = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])

    center_x, center_y, center_z = position[0], position[1], position[2]
    curr_x, curr_y, curr_z = 0, 0, radius
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