import numpy as np
import pandas as pd
from models import model_AB
from euler_animate import animate
import matplotlib.pyplot as plt
import config
from typing import Callable


def euler(func: Callable, 
          start_vec: tuple[float, float, float, float, float, float, float, float, float, float, float, float], 
          h: float, 
          tau: np.ndarray, 
          A: float, 
          B: float, 
          t_final: int, 
          save: bool = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform euler's method by repeatedly calling the given function. Note that this function
    is programmed to handle systems of 12 equations (4 position vectors in succession)

    start_vec: initial vector of cells
    h: step size for euler
    tau: times to evaluate ODE at (in [0,1])
    A: parameter of ODE
    B: parameter of ODE
    t_final: last timestamp in raw data (time of last step). Used to divide through so that tau is in [0,1]
    save: if True, saves data, else doesn't (useful when you don't want the optimization process to save)
    """


    curr_pos = start_vec

    # initial position (t=0)
    euler_coords = [curr_pos]
    distance = {"12": [1], 
                "13": [1], 
                "14": [1 * np.sqrt(2)],
                "23": [1 * np.sqrt(2)], 
                "24": [1], 
                "34": [1]}
    angle = {"dorsal1": [90], 
             "dorsal2": [90], 
             "anterior1": [0],
             "anterior2": [0]}

    # diagonal springs on or not
    diag14 = False
    diag23 = False

    # take "steps" steps
    for time in tau:
        # turn on the spring
        if distance["14"][-1] <= 1:
            diag14 = True
        if distance["23"][-1] <= 1:
            diag23 = True
        # perform euler's method
        curr_velocity = func(curr_pos, A, B, time, t_final, diag14, diag23)
        curr_pos = curr_pos + h * curr_velocity
        euler_coords.append(curr_pos)
        #position vectors
        position_1 = np.array([curr_pos[0], curr_pos[1], curr_pos[2]])
        position_2 = np.array([curr_pos[3], curr_pos[4], curr_pos[5]])
        position_3 = np.array([curr_pos[6], curr_pos[7], curr_pos[8]])
        position_4 = np.array([curr_pos[9], curr_pos[10], curr_pos[11]])
        # axis vectors
        axis_1 = position_1-position_3
        axis_2 = position_2-position_4
        anterior_ax_1 = np.array([axis_1[1], axis_1[2]])
        anterior_ax_2 = np.array([axis_2[1], axis_2[2]])
        dorsal_ax_1 = np.array([axis_1[0], axis_1[1]])
        dorsal_ax_2 = np.array([axis_2[0], axis_2[1]])
        # location of "0" degrees, used to dot with axis vectors to obtain angle
        dorsal_0 = np.array([1,0])
        anterior_0 = np.array([1,0])

        # dot product formula to obtain angle
        dorsal_ang_1 = np.arccos(np.dot(dorsal_ax_1, dorsal_0)/(np.linalg.norm(dorsal_ax_1)))
        dorsal_ang_2 = np.arccos(np.dot(dorsal_ax_2, dorsal_0)/(np.linalg.norm(dorsal_ax_2)))
        anterior_ang_1 = np.arccos(np.dot(anterior_ax_1, anterior_0)/(np.linalg.norm(anterior_ax_1)))
        anterior_ang_2 = np.arccos(np.dot(anterior_ax_2, anterior_0)/(np.linalg.norm(anterior_ax_2)))

        # checks quadrant of axis vector
        if dorsal_ax_1[1] < 0:
            dorsal_ang_1 *= -1
        if dorsal_ax_2[1] < 0:
            dorsal_ang_2 *= -1
        if anterior_ax_1[1] < 0:
            anterior_ang_1 *= -1
        if anterior_ax_2[1] < 0:
            anterior_ang_2 *= -1

        # convert to radians and saved
        angle["dorsal1"].append(dorsal_ang_1 * (180 / np.pi))
        angle["dorsal2"].append(dorsal_ang_2 * (180 / np.pi))
        angle["anterior1"].append(anterior_ang_1 * (180 / np.pi))
        angle["anterior2"].append(anterior_ang_2 * (180 / np.pi))

        # compute Euclidean distance and save
        distance["12"].append(np.linalg.norm(np.subtract(position_1, position_2)))
        distance["13"].append(np.linalg.norm(np.subtract(position_1, position_3)))
        distance["14"].append(np.linalg.norm(np.subtract(position_1, position_4)))
        distance["23"].append(np.linalg.norm(np.subtract(position_2, position_3)))
        distance["24"].append(np.linalg.norm(np.subtract(position_2, position_4)))
        distance["34"].append(np.linalg.norm(np.subtract(position_3, position_4)))

    # convert into pd.DataFrame
    column_names=[str(col) for col in range(12)]
    euler_df = pd.DataFrame(data=euler_coords, columns=column_names)
    distance_df = pd.DataFrame(distance)
    angle_df = pd.DataFrame(angle)

    # separate 12 coordinate vectors into 4 position dataframes
    output_eulers = []
    for col in range(12):
        if col % 3 == 0:
            euler_subdf = euler_df[[str(col), str(col+1), str(col+2)]].copy()
            euler_subdf.rename(columns={str(col): "x", str(col+1): "y", str(col+2): "z"}, inplace=True)
            output_eulers.append(euler_subdf)

    # save data if desired
    if save:
        distance_df.to_csv(config.DISTANCE_DATAPATH)
        angle_df.to_csv(config.ANGLES_DATAPATH)
        output_eulers[0].to_csv(config.POSITION_1_DATAPATH)
        output_eulers[1].to_csv(config.POSITION_2_DATAPATH)
        output_eulers[2].to_csv(config.POSITION_3_DATAPATH)
        output_eulers[3].to_csv(config.POSITION_4_DATAPATH)

    return (output_eulers, distance_df, angle_df)




    
    