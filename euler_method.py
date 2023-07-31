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

    # initial position (t=0)
    curr_pos = start_vec
    euler_coords = [curr_pos]
    dist_ang = comp_dist_ang(curr_pos)
    distance = {"12": [dist_ang["distance"]["12"]], 
                "13": [dist_ang["distance"]["13"]], 
                "14": [dist_ang["distance"]["14"]],
                "23": [dist_ang["distance"]["23"]], 
                "24": [dist_ang["distance"]["24"]], 
                "34": [dist_ang["distance"]["34"]]}
    angle = {"dorsal1": [dist_ang["angle"]["dorsal1"]], 
             "dorsal2": [dist_ang["angle"]["dorsal2"]], 
             "anterior1": [dist_ang["angle"]["anterior1"]],
             "anterior2": [dist_ang["angle"]["anterior2"]]}

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
        curr_velocity = func(curr_pos, A, B, time, diag14, diag23)
        curr_pos = curr_pos + h * curr_velocity
        euler_coords.append(curr_pos)
        dist_ang = comp_dist_ang(curr_pos)

        # convert to radians and saved
        angle["dorsal1"].append(dist_ang["angle"]["dorsal1"])
        angle["dorsal2"].append(dist_ang["angle"]["dorsal2"])
        angle["anterior1"].append(dist_ang["angle"]["anterior1"])
        angle["anterior2"].append(dist_ang["angle"]["anterior2"])

        # compute Euclidean distance and save
        distance["12"].append(dist_ang["distance"]["12"])
        distance["13"].append(dist_ang["distance"]["13"])
        distance["14"].append(dist_ang["distance"]["14"])
        distance["23"].append(dist_ang["distance"]["23"])
        distance["24"].append(dist_ang["distance"]["24"])
        distance["34"].append(dist_ang["distance"]["34"])

    # convert into pd.DataFrame
    column_names=[str(col) for col in range(12)]
    euler_df = pd.DataFrame(data=euler_coords, columns=column_names)
    distance_df = pd.DataFrame(distance)
    angle_df = pd.DataFrame(angle)

    # separate 12 coordinate vectors into 4 position dataframes
    position_eulers = []
    for col in range(12):
        if col % 3 == 0:
            temp_subdf = euler_df[[str(col), str(col+1), str(col+2)]].copy()
            temp_subdf.rename(columns={str(col): "x", str(col+1): "y", str(col+2): "z"}, inplace=True)
            position_eulers.append(temp_subdf)

    # save data if desired
    if save:
        save_euler(angle_df, distance_df, position_eulers)

    return (position_eulers, distance_df, angle_df)



def comp_dist_ang(curr_pos: tuple[float, float, float, float, float, float, float, float, float, float, float, float]):
    """
    Computes and returns the distance and phi and theta in degrees.

    curr_pos: 12 dimensional vector representing the 4 position vectors of 4 cells 
    """
    output = {"distance": {}, "angle": {}}

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

    # compute Euclidean distance and save
    output["distance"]["12"] = np.linalg.norm(np.subtract(position_1, position_2))
    output["distance"]["13"] = np.linalg.norm(np.subtract(position_1, position_3))
    output["distance"]["14"] = np.linalg.norm(np.subtract(position_1, position_4))
    output["distance"]["23"] = np.linalg.norm(np.subtract(position_2, position_3))
    output["distance"]["24"] = np.linalg.norm(np.subtract(position_2, position_4))
    output["distance"]["34"] = np.linalg.norm(np.subtract(position_3, position_4))

    # convert to radians and saved
    output["angle"]["dorsal1"] = dorsal_ang_1 * (180 / np.pi)
    output["angle"]["dorsal2"] = dorsal_ang_2 * (180 / np.pi)
    output["angle"]["anterior1"] = anterior_ang_1 * (180 / np.pi)
    output["angle"]["anterior2"] = anterior_ang_2 * (180 / np.pi)

    return output
    
    
def save_euler(angle_df: pd.DataFrame, distance_df: pd.DataFrame, position_eulers: tuple[pd.DataFrame,
                                                                                       pd.DataFrame,
                                                                                       pd.DataFrame,
                                                                                       pd.DataFrame]
                                                                                       ) -> None:
    """
    Saves the distance, angle, and position data from euler's method. 
    """
    distance_df.to_csv(config.DISTANCE_DATAPATH)
    angle_df.to_csv(config.ANGLES_DATAPATH)
    position_eulers[0].to_csv(config.POSITION_1_DATAPATH)
    position_eulers[1].to_csv(config.POSITION_2_DATAPATH)
    position_eulers[2].to_csv(config.POSITION_3_DATAPATH)
    position_eulers[3].to_csv(config.POSITION_4_DATAPATH)