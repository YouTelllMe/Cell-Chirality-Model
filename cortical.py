import pandas as pd
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
import config
from enum import Enum, auto
from dataclasses import dataclass

@dataclass
class CorticalFlowFit(Enum):
    LINEAR = auto()
    QUADRATIC = auto()


def fit_func(x, time, data, fit_type):
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


if __name__ == "__main__":

    # read cortical file; drop first row (column index) and reset row index
    corticalflow_xls = pd.ExcelFile(config.CORTICALFLOW_PATH)
    corticalflow = pd.read_excel(corticalflow_xls, "corticalflow")
    corticalflow = corticalflow.drop(index=[0]).reset_index(drop=True)

    # get time column and drop time column from data
    time = corticalflow["Time (s)"]
    corticalflow = corticalflow.drop(columns=["Time (s)"])
    # N = 10; split into left and right side; absolute value negative data for averaging
    corticalflow_right = corticalflow.iloc[:, 0:10]
    corticalflow_left = corticalflow.iloc[:, 10:20]
    corticalflow_left = corticalflow_left.apply(lambda x: abs(x))
    
    # fit data using αte^(-λt)
    alpha_r, lambda_r = scipy.optimize.fmin(fit_func, [1, 1], args=(time, corticalflow_right, CorticalFlowFit.LINEAR))
    alpha_l, lambda_l = scipy.optimize.fmin(fit_func, [1, 1], args=(time, corticalflow_left, CorticalFlowFit.LINEAR))

    # average angles
    cortical_sum_right = 0
    cortical_sum_left = 0
    for column in range(10):
        cortical_sum_right += corticalflow_right.iloc[:,column].to_numpy()
        cortical_sum_left += corticalflow_left.iloc[:,column].to_numpy()
    cortical_average_right = cortical_sum_right / 10
    cortical_average_left = cortical_sum_left / 10

    fig, ((axLeft, axRight), (axLeftSq, axRightSq)) = plt.subplots(2, 2)

    # axLeftSq.plot(time, cortical_average_left)
    # axLeftSq.plot(time, np.multiply(alpha_l_sq*time**2,np.e**(-lambda_l_sq * time)))
    # axRightSq.plot(time, cortical_average_right)
    # axRightSq.plot(time, np.multiply(alpha_r_sq*time**2,np.e**(-lambda_r_sq * time)))

    axLeft.plot(time, cortical_average_left)
    axLeft.plot(time, np.multiply(alpha_l*time,np.e**(-lambda_l * time)))
    axLeft.plot(time, np.multiply(0.000527*time,np.e**(-0.01466569 * time)))
    axRight.plot(time, cortical_average_right)
    axRight.plot(time, np.multiply(alpha_r*time,np.e**(-lambda_r * time)))
    axRight.plot(time, np.multiply(0.000527*time,np.e**(-0.01466569 * time)))
    plt.savefig("cortical_fit.png")

    # print average fit coefficients 
    print((alpha_r+alpha_l)/2, (lambda_r+lambda_l)/2)
