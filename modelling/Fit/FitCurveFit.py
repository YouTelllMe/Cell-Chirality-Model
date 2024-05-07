import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
from collections.abc import Sequence
from ..Simulator import Simulator
from ..ModelAB import ModelAB


def fit_model_whole(raw_data):
    """
    """

    (dorsal_anterior, 
    dorsal_posterior, 
    dorsal_t,
    anterior_anterior, 
    anterior_dorsal, 
    anterior_t) = raw_data

    manual_distances = np.ones(160)
    dorsal_anterior = dorsal_anterior.to_numpy().flatten("F")
    dorsal_posterior = dorsal_posterior.to_numpy().flatten("F")
    anterior_anterior = anterior_anterior.to_numpy().flatten("F")
    anterior_dorsal = anterior_dorsal.to_numpy().flatten("F")

    y_data = np.concatenate((dorsal_anterior, dorsal_posterior, anterior_anterior, anterior_dorsal, manual_distances))
    x_data = ()

    popt, pcov = curve_fit(fit_model_curve, x_data, y_data, p0=(20,1))

    alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
    n = len(y_data)    # number of data points
    p = len(popt) # number of parameters
    df = max(0, n - p) # number of degrees of freedom
    tval = t.ppf(1.0-alpha/2., df) # student-t value for the df and confidence level

    return (popt, pcov, ((np.diag(pcov)[0]**0.5)*tval,
                         (np.diag(pcov)[1]**0.5)*tval))


def fit_model_curve(x: Sequence[float], A: float, B: float):
    """
    """
    print(A, B)
    sim = Simulator(ModelAB, (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5), A=A, B=B, t_final=195)
    sim.run(False)

    dorsal_anterior = sim.angle["dorsal_ABa"]
    dorsal_posterior = sim.angle["dorsal_ABp"]
    anterior_anterior = sim.angle["anterior_ABa"]
    anterior_dorsal = sim.angle["anterior_ABp"]

    
    computed_instance_N = []
    for angle_type in (dorsal_anterior, dorsal_posterior, anterior_anterior, anterior_dorsal):
        for _ in range(10):
            computed_instance_N = np.concatenate((computed_instance_N, angle_type))

    computed_instance_N = np.concatenate((computed_instance_N, 
                                          sim.distance["12"],
                                          sim.distance["13"],
                                          sim.distance["24"],
                                          sim.distance["34"]))
    return computed_instance_N



# def fit_model_wholeAB():
#     """
#     """

#     (dorsal_anterior, 
#     dorsal_posterior, 
#     dorsal_t,
#     anterior_anterior, 
#     anterior_dorsal, 
#     anterior_t) = DataProcessing.get_data()

#     manual_distances = np.ones(160)
#     dorsal_anterior = dorsal_anterior.to_numpy().flatten("F")
#     dorsal_posterior = dorsal_posterior.to_numpy().flatten("F")
#     anterior_anterior = anterior_anterior.to_numpy().flatten("F")
#     anterior_dorsal = anterior_dorsal.to_numpy().flatten("F")

#     y_data = np.concatenate((dorsal_anterior, dorsal_posterior, anterior_anterior, anterior_dorsal, manual_distances))
#     x_data = ()

#     popt, pcov = curve_fit(fit_model_curveAB, x_data, y_data, p0=config.GUESSAB)

#     alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
#     n = len(y_data)    # number of data points
#     p = len(popt) # number of parameters
#     df = max(0, n - p) # number of degrees of freedom
#     tval = t.ppf(1.0-alpha/2., df) # student-t value for the df and confidence level

#     return (popt, pcov, ((np.diag(pcov)[0]**0.5)*tval,
#                          (np.diag(pcov)[1]**0.5)*tval))


# def fit_model_curveAB(x: Sequence[float], A: float, B: float):
#     """
#     """

#     euler_data = Euler(ModelAB(A, B, 
#             FourCellSystem(
#                 Cell((-0.5,0.5,0.5)), 
#                 Cell((0.5,0.5,0.5)), 
#                 Cell((-0.5,-0.5,0.5)), 
#                 Cell((0.5,-0.5,0.5))
#             ))
#     ).run(False)
    
        
#     indicies = range(0, config.MODEL_STEPS, config.STEP_SCALE)
#     distance_df = euler_data[1].iloc[indicies].reset_index(drop=True)
#     angle_df = euler_data[2].iloc[indicies].reset_index(drop=True)
#     epsilon = 1

#     dorsal_anterior = angle_df["dorsal2"]
#     dorsal_posterior = angle_df["dorsal1"]
#     anterior_anterior = angle_df["anterior2"]
#     anterior_dorsal = angle_df["anterior1"]

    
#     computed_instance_N = []
#     for angle_type in (dorsal_anterior, dorsal_posterior, anterior_anterior, anterior_dorsal):
#         for _ in range(config.DATA_N):
#             computed_instance_N = np.concatenate((computed_instance_N, angle_type))

#     computed_instance_N = np.concatenate((computed_instance_N, 
#                                           distance_df["12"],
#                                           distance_df["13"],
#                                           distance_df["24"],
#                                           distance_df["34"]))
#     return computed_instance_N
