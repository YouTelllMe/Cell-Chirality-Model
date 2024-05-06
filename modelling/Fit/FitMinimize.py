from Modeling.Cell import FourCellSystem, Cell
import old.config as config
import Modeling.Simulator as Simulator
import pandas as pd
import numpy as np
from Model import ModelABC
import DataProcessing
from scipy.optimize import fmin, minimize 

#TODO
def fit_fmin_model(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None) -> tuple[float, float]:
    """
    """

    if data is None:
        data = DataProcessing.get_data()
    (dorsal_anterior, 
    dorsal_posterior, 
    dorsal_t,
    anterior_anterior, 
    anterior_dorsal, 
    anterior_t) = data
        
    A, B, C = fmin(residual_squared, 
                config.GUESS, 
                args=(anterior_anterior, 
                    anterior_dorsal,
                    dorsal_anterior, 
                    dorsal_posterior, 
                    ))
    return (A, B, C)

#TODO
def fit_minimize_model(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None) -> tuple[float, float]:
    """
    """

    if data is None:
        data = DataProcessing.get_data()
    (dorsal_anterior, 
    dorsal_posterior, 
    dorsal_t,
    anterior_anterior, 
    anterior_dorsal, 
    anterior_t) = data
        
    A, B, C = minimize(residual_squared, 
                config.GUESS, 
                args=(anterior_anterior, 
                    anterior_dorsal,
                    dorsal_anterior, 
                    dorsal_posterior, 
                    ))
    return (A, B, C)


def residual_squared(x: tuple[float, float, float], 
        anterior_anterior: pd.DataFrame, 
        anterior_dorsal: pd.DataFrame, 
        dorsal_anterior: pd.DataFrame, 
        dorsal_posterior: pd.DataFrame):
    """
    residual_squares of angles
    """
    # initial point not included within tau
    A, B, C = x
    N = config.MODEL_STEPS
    # takes 400 steps (initial position vector inclusive)
    tau = np.linspace(1/N, 1, N-1)

    _, computed_distances, computed_angle = Simulator(ModelABC(A, B, C, 
            FourCellSystem(
                Cell((-0.5,0.5,0.5)), 
                Cell((0.5,0.5,0.5)), 
                Cell((-0.5,-0.5,0.5)), 
                Cell((0.5,-0.5,0.5))
            ))
    ).run(False)

    computed_data_index = range(0, config.MODEL_STEPS, config.STEP_SCALE)
    computed_dorsal_1 = computed_angle["dorsal1"][computed_data_index].to_numpy()
    computed_dorsal_2 = computed_angle["dorsal2"][computed_data_index].to_numpy()
    computed_anterior_1 = computed_angle["anterior1"][computed_data_index].to_numpy()
    computed_anterior_2 = computed_angle["anterior2"][computed_data_index].to_numpy()
    residual_square = 0
    for column_index in range(config.DATA_N):
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
    residual_square += epsilon * ((np.sum(computed_distances["12"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["24"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["34"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["13"].to_numpy() - np.ones(400))) ** 2)
    print(residual_square)
    return residual_square