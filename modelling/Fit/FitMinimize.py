import pandas as pd
import numpy as np
from scipy.optimize import fmin, minimize 
from ..ModelAB import ModelAB
from ..Simulator import Simulator

def fit_fmin_model(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None) -> tuple[float, float]:
    """
    """

    (dorsal_anterior, 
    dorsal_posterior, 
    dorsal_t,
    anterior_anterior, 
    anterior_dorsal, 
    anterior_t) = data
        
    A, B = fmin(residual_squared, 
                [1,1], 
                args=(anterior_anterior, 
                    anterior_dorsal,
                    dorsal_anterior, 
                    dorsal_posterior, 
                    ))
    return (A, B)


def residual_squared(x: tuple[float, float, float], 
        anterior_anterior: pd.DataFrame, 
        anterior_dorsal: pd.DataFrame, 
        dorsal_anterior: pd.DataFrame, 
        dorsal_posterior: pd.DataFrame):
    """
    residual_squares of angles
    """
    # initial point not included within tau
    A, B = x
    sim = Simulator(ModelAB, (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5), A=A, B=B, t_final=195)
    sim.run(False)

    computed_dorsal_1 = sim.angle["dorsal_ABa"].to_numpy()
    computed_dorsal_2 = sim.angle["dorsal_ABp"].to_numpy()
    computed_anterior_1 = sim.angle["anterior_ABa"].to_numpy()
    computed_anterior_2 = sim.angle["anterior_ABp"].to_numpy()
    residual_square = 0
    for column_index in range(10):
        da_residual = dorsal_anterior.iloc[:,column_index].to_numpy() - computed_dorsal_2
        dp_residual = dorsal_posterior.iloc[:,column_index].to_numpy() - computed_dorsal_1
        aa_residual = anterior_anterior.iloc[:,column_index].to_numpy() - computed_anterior_2
        ad_residual = anterior_dorsal.iloc[:,column_index].to_numpy() - computed_anterior_1
        
        residual_square += (np.linalg.norm(da_residual) ** 2
                          + np.linalg.norm(dp_residual) ** 2
                          + np.linalg.norm(aa_residual) ** 2
                          + np.linalg.norm(ad_residual) ** 2)

    epsilon = 1

    # makes distance 1
    residual_square += epsilon * ((np.sum(sim.distance["12"].to_numpy() - np.ones(40))) ** 2 +
                                (np.sum(sim.distance["24"].to_numpy() - np.ones(40))) ** 2 +
                                (np.sum(sim.distance["34"].to_numpy() - np.ones(40))) ** 2 +
                                (np.sum(sim.distance["13"].to_numpy() - np.ones(40))) ** 2)
    print(residual_square)
    return residual_square