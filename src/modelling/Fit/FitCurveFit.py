import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
from collections.abc import Sequence
from ..Simulator import Simulator
from .config import GET_VELOCITY, INIT


def fit_model_whole(raw_data):
    """
    """

    (ABa_dorsal, ABp_dorsal, dorsal_t, ABa_ant, ABp_ant, anterior_t) = raw_data

    manual_distances = np.ones(160)
    ABa_dorsal = ABa_dorsal.to_numpy().flatten("F")
    ABp_dorsal = ABp_dorsal.to_numpy().flatten("F")
    ABa_ant = ABa_ant.to_numpy().flatten("F")
    ABp_ant = ABp_ant.to_numpy().flatten("F")

    y_data = np.concatenate((ABa_dorsal, ABp_dorsal, ABa_ant, ABp_ant, manual_distances))
    x_data = ()

    popt, pcov = curve_fit(fit_model_curve, x_data, y_data, p0=(1,1))

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
    sim = Simulator(GET_VELOCITY(A, B, 195), INIT)
    sim.run(False)

    ABa_dorsal = sim.angle["ABa_dorsal"]
    ABp_dorsal = sim.angle["ABp_dorsal"]
    ABa_ant = sim.angle["ABa_anterior"]
    ABp_ant = sim.angle["ABp_anterior"]

    
    computed_instance_N = []
    for angle_type in (ABa_dorsal, ABp_dorsal, ABa_ant, ABp_ant):
        for _ in range(10):
            computed_instance_N = np.concatenate((computed_instance_N, angle_type))

    computed_instance_N = np.concatenate((computed_instance_N, 
                                          sim.distance["12"],
                                          sim.distance["23"],
                                          sim.distance["34"],
                                          sim.distance["14"]))
    return computed_instance_N

