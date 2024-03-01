import pandas as pd
import utils
import numpy as np
import config
from scipy.optimize import curve_fit
from scipy.stats import t
from Euler import Euler
from Cell import Cell, FourCellSystem
from collections.abc import Sequence
from Model.ModelABC import ModelABC



def fit_model_whole():
    """
    """

    (dorsal_anterior, 
    dorsal_posterior, 
    dorsal_t,
    anterior_anterior, 
    anterior_dorsal, 
    anterior_t) = utils.get_data()

    manual_distances = np.ones(160)
    dorsal_anterior = dorsal_anterior.to_numpy().flatten("F")
    dorsal_posterior = dorsal_posterior.to_numpy().flatten("F")
    anterior_anterior = anterior_anterior.to_numpy().flatten("F")
    anterior_dorsal = anterior_dorsal.to_numpy().flatten("F")

    y_data = np.concatenate((dorsal_anterior, dorsal_posterior, anterior_anterior, anterior_dorsal, manual_distances))
    x_data = ()

    popt, pcov = curve_fit(fit_model_curve, x_data, y_data, p0=config.GUESS)

    alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
    n = len(y_data)    # number of data points
    p = len(popt) # number of parameters
    df = max(0, n - p) # number of degrees of freedom
    tval = t.ppf(1.0-alpha/2., df) # student-t value for the df and confidence level

    return (popt, pcov, ((np.diag(pcov)[0]**0.5)*tval,
                         (np.diag(pcov)[1]**0.5)*tval))


def fit_model_curve(x: Sequence[float], A: float, B: float, C: float):
    """
    """

    euler_data = Euler(ModelABC(A, B, C, 
            FourCellSystem(
                Cell((-0.5,0.5,0.5)), 
                Cell((0.5,0.5,0.5)), 
                Cell((-0.5,-0.5,0.5)), 
                Cell((0.5,-0.5,0.5))
            ))
    ).run(False)
    
        
    indicies = range(0, config.MODEL_STEPS, config.STEP_SCALE)
    distance_df = euler_data[1].iloc[indicies].reset_index(drop=True)
    angle_df = euler_data[2].iloc[indicies].reset_index(drop=True)
    epsilon = 1

    dorsal_anterior = angle_df["dorsal2"]
    dorsal_posterior = angle_df["dorsal1"]
    anterior_anterior = angle_df["anterior2"]
    anterior_dorsal = angle_df["anterior1"]

    
    computed_instance_N = []
    for angle_type in (dorsal_anterior, dorsal_posterior, anterior_anterior, anterior_dorsal):
        for _ in range(config.DATA_N):
            computed_instance_N = np.concatenate((computed_instance_N, angle_type))

    computed_instance_N = np.concatenate((computed_instance_N, 
                                          distance_df["12"],
                                          distance_df["13"],
                                          distance_df["24"],
                                          distance_df["34"]))
    return computed_instance_N
