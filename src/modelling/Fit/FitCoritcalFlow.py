from dataclasses import dataclass
from enum import Enum, auto
from collections.abc import Sequence
import pandas as pd 
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt




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


def fit_cortical(cortical_data) -> None:
    """
    Fit cortical data to αte^(-λt)
    """

    # process raw_data, absolute value negative cortical flow for fitting
    (corticalflow_right, corticalflow_left, time) = cortical_data
    corticalflow_left = corticalflow_left.apply(lambda x: abs(x))
    
    # fit data using αte^(-λt)
    alpha_r, lambda_r = fmin(fit_cortical_func, [1, 1], args=(time, corticalflow_right, CorticalFlowFit.LINEAR))
    alpha_l, lambda_l = fmin(fit_cortical_func, [1, 1], args=(time, corticalflow_left, CorticalFlowFit.LINEAR))

    # average angles
    cortical_average_right = column_average(corticalflow_right)
    cortical_average_left = column_average(corticalflow_left)

    # initialize figure and plot
    fig, (axLeft, axRight) = plt.subplots(1, 2)
    axLeft.plot(time, cortical_average_left)
    axLeft.plot(time, np.multiply(alpha_l*time,np.e**(-lambda_l * time)))
    axRight.plot(time, cortical_average_right)
    axRight.plot(time, np.multiply(alpha_r*time,np.e**(-lambda_r * time)))
    plt.savefig("cortical.png")

    # print average fit coefficients 
    print(alpha_r, lambda_r)
    print(alpha_l, lambda_l)


def column_average(df: pd.DataFrame) -> np.ndarray:
    """
    return a column-wise average of a pandas dataframe
    """
    col_sum = 0
    num_cols = len(df.columns)
    for column in range(num_cols):
        col_sum += df.iloc[:,column].to_numpy()
    col_average = col_sum / num_cols
    return col_average