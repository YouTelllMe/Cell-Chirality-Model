import os
import pandas as pd
import scipy.optimize
from fit import fit
import matplotlib.pyplot as plt
import numpy as np
from euler_method import euler
from models import model_AB
from euler_animate import animate
import config
import utils

def run_euler():
    # initial point not included within tau
    t_final = 195
    A, B = 6.739499413217986, 0.040047325767577635
    N = 400
    # takes 399 steps (initial position vector is given)
    tau = np.linspace(1/N, 1, N-1)
    euler_df = euler(model_AB, np.array([-0.5,0.5,0.5, 0.5,0.5,0.5, -0.5,-0.5,0.5, 0.5,-0.5,0.5]), 1/N, tau, A, B, t_final)[0]
    animate(euler_df)
    plt.show()


def fit_model():
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
    print(A, B) #6.412403028415063 0.03803481931476198

    # average angles for plotting
    da_average = utils.column_average(dorsal_anterior)
    dp_average = utils.column_average(dorsal_posterior)
    aa_averege = utils.column_average(anterior_anterior)
    ad_average = utils.column_average(anterior_dorsal)


    computed_angle = pd.read_csv(config.ANGLES_DATAPATH)
    computed_data_index = range(0, 400, 10)
    computed_dorsal_1 = computed_angle["dorsal1"][computed_data_index].to_numpy()
    computed_dorsal_2 = computed_angle["dorsal2"][computed_data_index].to_numpy()
    computed_anterior_1 = computed_angle["anterior1"][computed_data_index].to_numpy()
    computed_anterior_2 = computed_angle["anterior2"][computed_data_index].to_numpy()

    fig, ((axDA, axDP),(axAA, axAD)) = plt.subplots(2, 2)

    axDA.plot(dorsal_t, computed_dorsal_2, label="model", markersize=2)
    axDA.plot(dorsal_t, da_average, label="average data", markersize=2)

    axDP.plot(dorsal_t, computed_dorsal_1, label="model", markersize=2)
    axDP.plot(dorsal_t, dp_average, label="average data", markersize=2)

    axAA.plot(anterior_t, computed_anterior_2, label="model", markersize=2)
    axAA.plot(anterior_t, aa_averege, label="average data", markersize=2)

    axAD.plot(anterior_t, computed_anterior_1, label="model", markersize=2)
    axAD.plot(anterior_t, ad_average, label="average data", markersize=2)

    axDA.title.set_text("Dorsal Angle - Anterior Axis")
    axDP.title.set_text("Dorsal Angle - Posterior Axis")
    axAA.title.set_text("Anterior Angle - Anterior Axis")
    axAD.title.set_text("Anterior Angle - Posterior Axis")

    fig.set_figheight(7)
    fig.set_figwidth(15)
    axDA.legend()
    axDP.legend()
    axAA.legend()
    axAD.legend()

    plt.savefig("fit.png")

if __name__ == "__main__":
    # fit_model()
    run_euler()
    # print("hi")