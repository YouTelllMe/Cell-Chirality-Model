import numpy as np
import pandas as pd
from models import model_AB
from euler_animate import animate
import matplotlib.pyplot as plt


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

    diag14 = False
    diag23 = False
    # take "steps" steps
    for time in tau:

        # turn on the spring
        if distance["14"][-1] <= 1:
            diag14 = True
        if distance["23"][-1] <= 1:
            diag23 = True

        curr_velocity = func(curr_pos, A, B, time, t_final, diag14, diag23)
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




    
    