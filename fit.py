import config
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
import scipy
import utils
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from euler_method import euler
from models import model_AB



def fit(x, anterior_anterior, anterior_dorsal, dorsal_anterior, dorsal_posterior):
    # initial point not included within tau
    t_final = 195
    A, B = x
    N = 400
    # takes 399 steps (initial position vector is directly added during euler)
    tau = np.linspace(1/N, 1, N-1)

    euler_data = euler(model_AB, 
                       np.array([-0.5,0.5,0.5, 0.5,0.5,0.5, -0.5,-0.5,0.5, 0.5,-0.5,0.5]), 
                       1/N, 
                       tau, 
                       A, 
                       B, 
                       t_final, 
                       False)
    
    computed_distances = euler_data[1]
    computed_angle = euler_data[2]

    computed_data_index = range(0, 400, 10)
    computed_dorsal_1 = computed_angle["dorsal1"][computed_data_index].to_numpy()
    computed_dorsal_2 = computed_angle["dorsal2"][computed_data_index].to_numpy()
    computed_anterior_1 = computed_angle["anterior1"][computed_data_index].to_numpy()
    computed_anterior_2 = computed_angle["anterior2"][computed_data_index].to_numpy()
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
    residual_square += epsilon * ((np.sum(computed_distances["12"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["24"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["34"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["13"].to_numpy() - np.ones(400))) ** 2)
    print(residual_square)
    return residual_square


@dataclass
class CorticalFlowFit(Enum):
    LINEAR = auto()
    QUADRATIC = auto()


def fit_cortical_func(x: tuple[float, float], time: Sequence, data: pd.DataFrame, fit_type: CorticalFlowFit) -> float:
    """
    Fit function that calculates the residuals of the cortical flow model to the raw data at a set of 
    parameters; used for fitting. 

    Returns the residual squared score
    """

    alpha, lam = x
    time = np.array(time)

    if fit_type == CorticalFlowFit.LINEAR:
        # αte^(-λt)
        output = alpha * time * np.e**(-lam * time)
    elif fit_type == CorticalFlowFit.QUADRATIC:
        # αt^2e^(-λt)
        output = alpha * time**2 * np.e**(-lam * time)

    residual_squared = 0 
    for column_index in range(10):
        residual_squared += np.linalg.norm(data.iloc[:, column_index] - output) ** 2
    
    return residual_squared


def fit_cortical():
    # read cortical file; drop first row (column index) and reset row index
    corticalflow_xls = pd.ExcelFile(config.CORTICALFLOW_PATH)
    corticalflow = pd.read_excel(corticalflow_xls, "corticalflow")

    corticalflow_right, corticalflow_left, time = utils.process_rawdf(corticalflow, "Time (s)")
    corticalflow_left = corticalflow_left.apply(lambda x: abs(x))
    
    # fit data using αte^(-λt)
    alpha_r, lambda_r = scipy.optimize.fmin(fit_cortical_func, [1, 1], args=(time, corticalflow_right, CorticalFlowFit.LINEAR))
    alpha_l, lambda_l = scipy.optimize.fmin(fit_cortical_func, [1, 1], args=(time, corticalflow_left, CorticalFlowFit.LINEAR))

    # average angles
    cortical_average_right = utils.column_average(corticalflow_right)
    cortical_average_left = utils.column_average(corticalflow_left)

    fig, (axLeft, axRight) = plt.subplots(1, 2)

    axLeft.plot(time, cortical_average_left)
    axLeft.plot(time, np.multiply(alpha_l*time,np.e**(-lambda_l * time)))
    axLeft.plot(time, np.multiply(0.000527*time,np.e**(-0.01466569 * time)))
    axRight.plot(time, cortical_average_right)
    axRight.plot(time, np.multiply(alpha_r*time,np.e**(-lambda_r * time)))
    axRight.plot(time, np.multiply(0.000527*time,np.e**(-0.01466569 * time)))
    plt.savefig(config.PLOT_FIT_CORTICAL)

    # print average fit coefficients 
    print((alpha_r+alpha_l)/2, (lambda_r+lambda_l)/2)


