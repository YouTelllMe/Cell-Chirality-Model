import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
from collections.abc import Sequence
from ..Simulator import Simulator
from .fit_config import GET_VELOCITY, INIT


def fit_model_whole(data):
    """
    """

    manual_distances = np.ones(160)
    ABa_dorsal = data["ABa_dorsal_avg"].to_numpy().flatten("F")
    ABp_dorsal = data["ABp_dorsal_avg"].to_numpy().flatten("F")
    ABa_ant = data["ABa_ant_avg"].to_numpy().flatten("F")
    ABp_ant = data["ABp_ant_avg"].to_numpy().flatten("F")

    y_data = np.concatenate((ABa_dorsal, ABp_dorsal, ABa_ant, ABp_ant, manual_distances))
    popt, pcov = curve_fit(fit_model_curve, (), y_data, p0=(0.1, 0.01), bounds=(0, np.inf))

    #TODO, this needs to be fixed; using average to fit now
    alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
    n = len(y_data)    # number of data points
    p = len(popt) # number of parameters
    df = max(0, n - p) # number of degrees of freedom
    tval = t.ppf(1.0-alpha/2., df) # student-t value for the df and confidence level

    return (popt, pcov, ((np.diag(pcov)[0]**0.5)*tval,
                         (np.diag(pcov)[1]**0.5)*tval))


def fit_model_curve(x: Sequence[float], *params):
    """
    """
    sim = Simulator(GET_VELOCITY(params), INIT)
    sim.run(False)

    return np.concatenate((sim.angle["ABa_dorsal"], 
                            sim.angle["ABp_dorsal"],
                            sim.angle["ABa_anterior"],
                            sim.angle["ABp_anterior"],
                            sim.distance["12"],
                            sim.distance["23"],
                            sim.distance["34"],
                            sim.distance["14"]))