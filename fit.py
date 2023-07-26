import numpy as np
import pandas as pd 
import os
from euler import euler, physics_system
import scipy
import matplotlib.pyplot as plt


# 40 data points, took 39 steps
# STEPS = 39

def fit(x, anterior_anterior, anterior_dorsal, dorsal_anterior, dorsal_posterior
        , anterior_t, dorsal_t):
    A, l, ck = x

    # l is not really l but arccos(l)*180/pi
    # get computed values
    # tau = np.linspace(0, 39, 40)
    # tau = ck * tau
    # computed_angle = euler(physics_system, np.array([-1,1,1, 1,1,1, -1,-1,1, 1,-1,1]), 0.0005, A, tau)[2]
    # computed_dorsal_1 = computed_angle["dorsal1"] * l
    # computed_dorsal_2 = computed_angle["dorsal2"] * l
    # computed_anterior_1 = computed_angle["anterior1"] * l
    # computed_anterior_2 = computed_angle["anterior2"] * l

    tau = np.linspace(0, 39, 40)
    tau = ck * tau
    computed_angle = euler(physics_system, np.array([-1,1,1, 1,1,1, -1,-1,1, 1,-1,1]), 0.0005, A, tau, l)[2]
    computed_dorsal_1 = computed_angle["dorsal1"]
    computed_dorsal_2 = computed_angle["dorsal2"]
    computed_anterior_1 = computed_angle["anterior1"]
    computed_anterior_2 = computed_angle["anterior2"]

    least_squares = 0
    for column_index in range(10):
        da_difference = dorsal_anterior.iloc[:,column_index] - computed_dorsal_2
        dp_difference = dorsal_posterior.iloc[:,column_index] - computed_dorsal_1
        aa_difference = anterior_anterior.iloc[:,column_index] - computed_anterior_2
        ad_difference = anterior_dorsal.iloc[:,column_index] - computed_anterior_1

        least_squares += (np.linalg.norm(da_difference.to_numpy())
                          + np.linalg.norm(dp_difference.to_numpy())
                          + np.linalg.norm(aa_difference.to_numpy())
                          + np.linalg.norm(ad_difference.to_numpy()))
    
    least_squares += np.linalg.norm(anterior_t - tau)
    least_squares += np.linalg.norm(dorsal_t - tau)
    

    print(least_squares)
    return least_squares


if __name__ == "__main__":
    current_dir = os.getcwd()
    # read xlsx files
    dorsal_xls = pd.ExcelFile(os.path.join(current_dir, "dorsal.xlsx"))
    anterior_xls = pd.ExcelFile(os.path.join(current_dir, "anterior.xlsx"))
    # remove first two rows
    dorsal = pd.read_excel(dorsal_xls, "dorsal").drop(index=[0]).reset_index(drop=True)
    anterior = pd.read_excel(anterior_xls, "anterior").drop(index=[0]).reset_index(drop=True)

    # take time column
    dorsal_t = dorsal["Time(s)"].to_numpy()
    anterior_t = anterior["Time(s)"].to_numpy()
    # drop time column
    dorsal.drop(columns=["Time(s)"], inplace=True)
    anterior.drop(columns=["Time(s)"], inplace=True)
    assert len(dorsal.columns) == 20, "data format incorrect"

    # dorsal_anterior is 24 (axis 2), dorsal_posterior is 13 (axis 1)
    dorsal_anterior = dorsal.iloc[:,0:10]
    dorsal_posterior = dorsal.iloc[:,10:20]
    # anterior_anterior is 24 (axis 2), anterior_dorsal is 13 (axis 1)
    anterior_anterior = anterior.iloc[:,0:10]
    anterior_dorsal = anterior.iloc[:,10:20]
        
    A, l, ck = scipy.optimize.fmin(fit, 
                                   [0.028502687016652727, 2.4330065244796355, 5.000000365235591], 
                                   args=(anterior_anterior, 
                                         anterior_dorsal,
                                         dorsal_anterior, 
                                         dorsal_posterior, 
                                         anterior_t, 
                                         dorsal_t,))
    print(A, l, ck)
    # average angles for plotting
    da_sum = dorsal_anterior.iloc[:,0].to_numpy()
    dp_sum = dorsal_posterior.iloc[:,0].to_numpy()
    aa_sum = anterior_anterior.iloc[:,0].to_numpy()
    ad_sum = anterior_dorsal.iloc[:,0].to_numpy()
    for column in range(1,10):
        da_sum += dorsal_anterior.iloc[:,column].to_numpy()
        dp_sum += dorsal_posterior.iloc[:,column].to_numpy()
        aa_sum += anterior_anterior.iloc[:,column].to_numpy()
        ad_sum += anterior_dorsal.iloc[:,column].to_numpy()
    da_average = da_sum / 10
    dp_average = dp_sum / 10
    aa_averege = aa_sum / 10
    ad_average = ad_sum / 10

    computed_angle = pd.read_csv(os.path.join(current_dir, "fit_angle.csv"))
    computed_dorsal_1 = computed_angle["dorsal1"]
    computed_dorsal_2 = computed_angle["dorsal2"]
    computed_anterior_1 = computed_angle["anterior1"]
    computed_anterior_2 = computed_angle["anterior2"]

    fig, ((axDA, axDP),(axAA, axAD)) = plt.subplots(2, 2)

    tau = np.linspace(0, 39, 40)
    tau = ck * tau
    axDA.plot(tau, computed_dorsal_2, label="model", markersize=2)
    axDA.plot(dorsal_t, da_average, label="average data", markersize=2)

    axDP.plot(tau, computed_dorsal_1, label="model", markersize=2)
    axDP.plot(dorsal_t, dp_average, label="average data", markersize=2)

    axAA.plot(tau, computed_anterior_2, label="model", markersize=2)
    axAA.plot(anterior_t, aa_averege, label="average data", markersize=2)

    axAD.plot(tau, computed_anterior_2, label="model", markersize=2)
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




