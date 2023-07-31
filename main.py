import config
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.optimize
import utils
from euler_animate import animate
from euler_method import euler
from fit import fit
from models import model_AB
from plot import plot_all

def run_euler(A: float, B: float) -> None:
    # initial point not included within tau
    t_final = 195
    # takes 399 steps (initial position vector is given)
    N = 400
    tau = np.linspace(1/N, 1, N-1)
    euler_df = euler(model_AB, 
                     np.array([-0.5,0.5,0.5, 0.5,0.5,0.5, -0.5,-0.5,0.5, 0.5,-0.5,0.5]), 
                     1/N, 
                     tau, 
                     A, 
                     B, 
                     t_final,
                     True)[0]
    animate(euler_df)
    plt.show()


def fit_model() -> tuple[float, float]:
    # read xlsx files
    dorsal_xls = pd.ExcelFile(config.DORSAL_ANGLE_PATH)
    anterior_xls = pd.ExcelFile(config.ANTERIOR_ANGLE_PATH)
    # remove first two rows
    dorsal = pd.read_excel(dorsal_xls, "dorsal")
    anterior = pd.read_excel(anterior_xls, "anterior")
    dorsal_anterior, dorsal_posterior, dorsal_t = utils.process_rawdf(dorsal, "Time(s)")
    anterior_anterior, anterior_dorsal, anterior_t = utils.process_rawdf(anterior, "Time(s)")
        
    A, B = scipy.optimize.fmin(fit, 
                              [6.309078133216052, 0.037810656687541716], 
                               args=(anterior_anterior, 
                                     anterior_dorsal,
                                     dorsal_anterior, 
                                     dorsal_posterior, 
                                     ))
    
    return (A, B)

if __name__ == "__main__":
    A, B = fit_model()
    run_euler(A, B)
    plot_all()