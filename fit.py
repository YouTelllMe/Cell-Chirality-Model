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
from scipy.stats import t


def fit_modelAB(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None) -> tuple[float, float]:
    """
    """

    if data is None:
        data = utils.get_data()
    (dorsal_anterior, 
    dorsal_posterior, 
    dorsal_t,
    anterior_anterior, 
    anterior_dorsal, 
    anterior_t) = data
        
    A, B = scipy.optimize.fmin(fit, 
                               config.GUESS, 
                               args=(anterior_anterior, 
                                     anterior_dorsal,
                                     dorsal_anterior, 
                                     dorsal_posterior, 
                                     ))
    return (A, B)


def fit_model_whole(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None):
    """
    """

    # takes 400 steps (initial position vector inclusive)
    N = config.MODEL_STEPS
    tau = np.linspace(1/N, 1, N-1)

    if data is None:
        data = utils.get_data()
    (dorsal_anterior, 
    dorsal_posterior, 
    dorsal_t,
    anterior_anterior, 
    anterior_dorsal, 
    anterior_t) = data

    manual_distances = np.ones(160)
    dorsal_anterior = dorsal_anterior.to_numpy().flatten("F")
    dorsal_posterior = dorsal_posterior.to_numpy().flatten("F")
    anterior_anterior = anterior_anterior.to_numpy().flatten("F")
    anterior_dorsal = anterior_dorsal.to_numpy().flatten("F")

    y_data = np.concatenate((dorsal_anterior, dorsal_posterior, anterior_anterior, anterior_dorsal, manual_distances))
    x_data = ()

    popt, pcov = scipy.optimize.curve_fit(fit_model_curve, x_data, y_data, p0=config.GUESS)

    alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
    n = len(y_data)    # number of data points
    p = len(popt) # number of parameters
    df = max(0, n - p) # number of degrees of freedom
    tval = t.ppf(1.0-alpha/2., df) # student-t value for the df and confidence level

    return (popt, pcov, ((np.diag(pcov)[0]**0.5)*tval,
                         (np.diag(pcov)[1]**0.5)*tval))

def fit(x: tuple[float, float], 
        anterior_anterior: pd.DataFrame, 
        anterior_dorsal: pd.DataFrame, 
        dorsal_anterior: pd.DataFrame, 
        dorsal_posterior: pd.DataFrame):
    """
    """
    # initial point not included within tau
    A, B = x
    N = config.MODEL_STEPS
    # takes 400 steps (initial position vector inclusive)
    tau = np.linspace(1/N, 1, N-1)

    euler_data = euler(model_AB, 
                       np.array([-0.5,0.5,0.5, 0.5,0.5,0.5, -0.5,-0.5,0.5, 0.5,-0.5,0.5]), 
                       1/N, 
                       tau, 
                       A, 
                       B, 
                       False)
    
    computed_distances = euler_data[1]
    computed_angle = euler_data[2]

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
    residual_square += epsilon * ((np.sum(computed_distances["12"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["24"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["34"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["13"].to_numpy() - np.ones(400))) ** 2)
    print(residual_square)
    return residual_square



def fit_model_curve(x: Sequence[float], A: float, B: float):
    """
    """
    # initial point not included within tau
    N = config.MODEL_STEPS
    # takes 400 steps (initial position vector inclusive)
    tau = np.linspace(1/N, 1, N-1)
    initial_vector = np.array([-0.5,0.5,0.5, 0.5,0.5,0.5, -0.5,-0.5,0.5, 0.5,-0.5,0.5])
    euler_data = euler(model_AB, 
                       initial_vector, 
                       1/N, 
                       tau, 
                       A, 
                       B, 
                       False)
    
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

if __name__ == "__main__":
    print(fit_model_whole(utils.get_data()))


#=======================================================================================================================#
"CORTICAL FLOW"

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
    for column_index in range(config.DATA_N):
        residual_squared += np.linalg.norm(data.iloc[:, column_index] - output) ** 2
    
    return residual_squared


def fit_cortical() -> None:
    """
    Fit cortical data to αte^(-λt)
    """

    # read cortical file; drop first row (column index) and reset row index
    corticalflow_xls = pd.ExcelFile(config.CORTICALFLOW_PATH)
    corticalflow = pd.read_excel(corticalflow_xls, "corticalflow")

    # process raw_data, absolute value negative cortical flow for fitting
    corticalflow_right, corticalflow_left, time = utils.process_rawdf(corticalflow, "Time (s)")
    corticalflow_left = corticalflow_left.apply(lambda x: abs(x))
    
    # fit data using αte^(-λt)
    alpha_r, lambda_r = scipy.optimize.fmin(fit_cortical_func, [1, 1], args=(time, corticalflow_right, CorticalFlowFit.LINEAR))
    alpha_l, lambda_l = scipy.optimize.fmin(fit_cortical_func, [1, 1], args=(time, corticalflow_left, CorticalFlowFit.LINEAR))

    # average angles
    cortical_average_right = utils.column_average(corticalflow_right)
    cortical_average_left = utils.column_average(corticalflow_left)

    # initialize figure and plot
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