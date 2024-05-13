import pandas as pd
import numpy as np
from scipy.optimize import fmin, minimize 
from ..Simulator import Simulator
from .config import GET_VELOCITY


def fit_fmin_model(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None) -> tuple[float, float]:
    """
    """

    (ABa_dorsal, ABp_dorsal, dorsal_t, ABa_ant, ABp_ant, anterior_t) = data
        
    A, B = fmin(residual_squared, 
                [0.12,10], 
                args=(ABa_dorsal, ABp_dorsal, ABa_ant, ABp_ant))
    return (A, B)


def residual_squared(x: tuple[float, float, float], 
        ABa_dorsal: pd.DataFrame, 
        ABp_dorsal: pd.DataFrame, 
        ABa_ant: pd.DataFrame, 
        ABp_ant: pd.DataFrame):
    """
    residual_squares of angles
    """
    # initial point not included within tau
    A, B = x
    # sim = Simulator(fun(A, B, 195), (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5), A=A, B=B, t_final=195,
    #                 surfaces=None)
    sim = Simulator(GET_VELOCITY(A, B, 195), (0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, -0.5, 0, -0.5, 0.5, 0))
    sim.run(False)

    computed_dorsal_ABa = sim.angle["ABa_dorsal"].to_numpy()
    computed_ABp_dorsal = sim.angle["ABp_dorsal"].to_numpy()
    computed_ABa_anterior = sim.angle["ABa_anterior"].to_numpy()
    computed_ABp_anterior = sim.angle["ABp_anterior"].to_numpy()
    residual_square = 0

    for column_index in range(10):
        da_residual = ABa_dorsal.iloc[:,column_index].to_numpy() - computed_dorsal_ABa
        dp_residual = ABp_dorsal.iloc[:,column_index].to_numpy() - computed_ABp_dorsal
        aa_residual = ABa_ant.iloc[:,column_index].to_numpy() - computed_ABa_anterior
        ap_residual = ABp_ant.iloc[:,column_index].to_numpy() - computed_ABp_anterior
        
        residual_square += (np.linalg.norm(da_residual) ** 2
                          + np.linalg.norm(dp_residual) ** 2
                          + np.linalg.norm(aa_residual) ** 2
                          + np.linalg.norm(ap_residual) ** 2)

    epsilon = 1

    # makes distance 1
    residual_square += epsilon * ((np.sum(sim.distance["12"].to_numpy() - np.ones(40))) ** 2 +
                                (np.sum(sim.distance["23"].to_numpy() - np.ones(40))) ** 2 +
                                (np.sum(sim.distance["34"].to_numpy() - np.ones(40))) ** 2 +
                                (np.sum(sim.distance["14"].to_numpy() - np.ones(40))) ** 2)
    
    print(residual_square)
    return residual_square