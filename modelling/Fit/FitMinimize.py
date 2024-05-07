import pandas as pd
import numpy as np
from scipy.optimize import fmin, minimize 
from ..ModelAB import ModelAB
from ..ModelExtendingSpring import ModelExtendingSpring
from ..Simulator import Simulator

def fit_fmin_model(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None) -> tuple[float, float]:
    """
    """

    (ABa_dorsal, ABp_dorsal, dorsal_t, ABa_ant, ABp_ant, anterior_t) = data
        
    A, B = fmin(residual_squared, 
                [1,1], 
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
    # sim = Simulator(ModelAB, (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5), A=A, B=B, t_final=195)
    sim = Simulator(ModelExtendingSpring, (0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, -0.5, 0, -0.5, 0.5, 0), 
            A=A, B=B, t_final=195, surfaces=[lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1])
    sim.run(False)

    dorsal_ABa = sim.angle["dorsal_ABa"].to_numpy()
    dorsal_ABp = sim.angle["dorsal_ABp"].to_numpy()
    anterior_ABa = sim.angle["anterior_ABa"].to_numpy()
    anterior_ABp = sim.angle["anterior_ABp"].to_numpy()
    residual_square = 0

    for column_index in range(10):
        da_residual = ABa_dorsal.iloc[:,column_index].to_numpy() - dorsal_ABa
        dp_residual = ABp_dorsal.iloc[:,column_index].to_numpy() - dorsal_ABp
        aa_residual = ABa_ant.iloc[:,column_index].to_numpy() - anterior_ABa
        ap_residual = ABp_ant.iloc[:,column_index].to_numpy() - anterior_ABp
        
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