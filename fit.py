import numpy as np
import pandas as pd 
import os
from euler import euler, physics_system
import scipy
import matplotlib.pyplot as plt


# 40 data points, took 39 steps
STEPS = 39

def fit(x, anterior_anterior, anterior_dorsal, dorsal_anterior, dorsal_posterior
        , anterior_t, dorsal_t):
    A, l, ck = x

    # l is not really l but arccos(l)*180/pi
    # get computed values
    computed_angle = euler(physics_system, np.array([-1,1,1, 1,1,1, -1,-1,1, 1,-1,1]), 0.0005, STEPS, A)[2]
    computed_dorsal_1 = computed_angle["dorsal1"] * l
    computed_dorsal_2 = computed_angle["dorsal2"] * l
    computed_anterior_1 = computed_angle["anterior1"] * l
    computed_anterior_2 = computed_angle["anterior2"] * l

    # set up tau
    tau = np.linspace(0, len(computed_angle)-1, len(computed_angle))
    tau = ck * tau


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
        
    optimized = scipy.optimize.fmin(fit, [1.5, 1, 1], args=(anterior_anterior, anterior_dorsal
                                                            ,dorsal_anterior, dorsal_posterior,
                                                            anterior_t, dorsal_t,))
    
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
    da_sum = da_sum / 10
    dp_sum = dp_sum / 10
    aa_sum = aa_sum / 10
    ad_sum = ad_sum / 10

    A = optimized[0]
    l = optimized[1]
    ck = optimized[2]
    print(A, l, ck)

    computed_angle = euler(physics_system, np.array([-1,1,1, 1,1,1, -1,-1,1, 1,-1,1]), 0.0005, STEPS, A)[2]
    computed_dorsal_1 = computed_angle["dorsal1"] * l
    computed_dorsal_2 = computed_angle["dorsal2"] * l
    computed_anterior_1 = computed_angle["anterior1"] * l
    computed_anterior_2 = computed_angle["anterior2"] * l
    tau = np.linspace(0, len(computed_angle)-1, len(computed_angle))
    tau = ck * tau

    fig, ((axDA, axDP),(axAA, axAD)) = plt.subplots(2, 2)

    axDA.plot(tau, computed_dorsal_2)
    axDA.plot(dorsal_t, da_sum)

    axDP.plot(tau, computed_dorsal_1)
    axDP.plot(dorsal_t, dp_sum)

    axAA.plot(tau, computed_anterior_2)
    axAA.plot(anterior_t, aa_sum)

    axAD.plot(tau, computed_anterior_2)
    axAD.plot(anterior_t, ad_sum)

    plt.show()



