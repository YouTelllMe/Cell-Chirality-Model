import DataProcessing
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def plot_fit():
    """
    Plot model fit against raw data. 
    """
    (dorsal_anterior, 
     dorsal_posterior, 
     dorsal_t,
     anterior_anterior, 
     anterior_posterior, 
     anterior_t) = DataProcessing.get_data()
    
    # average angles for plotting
    da_average = DataProcessing.column_average(dorsal_anterior)
    dp_average = DataProcessing.column_average(dorsal_posterior)
    aa_average = DataProcessing.column_average(anterior_anterior)
    ap_average = DataProcessing.column_average(anterior_posterior)

    # get standard error of the mean
    da_std = DataProcessing.get_std(dorsal_anterior, axis=1) / np.sqrt(config.DATA_STEPS)
    dp_std = DataProcessing.get_std(dorsal_posterior, axis=1) / np.sqrt(config.DATA_STEPS)
    aa_std = DataProcessing.get_std(anterior_anterior, axis=1) / np.sqrt(config.DATA_STEPS)
    ap_std = DataProcessing.get_std(anterior_posterior, axis=1) / np.sqrt(config.DATA_STEPS)

    # load data
    computed_angle = pd.read_csv(config.ANGLES_DATAPATH)
    computed_data_index = range(0, config.MODEL_STEPS, config.STEP_SCALE)
    computed_dorsal_1 = computed_angle["dorsal1"][computed_data_index].to_numpy()
    computed_dorsal_2 = computed_angle["dorsal2"][computed_data_index].to_numpy()
    computed_anterior_1 = computed_angle["anterior1"][computed_data_index].to_numpy()
    computed_anterior_2 = computed_angle["anterior2"][computed_data_index].to_numpy()

    #t-test
    da_np = dorsal_anterior.to_numpy()
    dp_np = dorsal_posterior.to_numpy()
    aa_np = anterior_anterior.to_numpy()
    ap_np = anterior_posterior.to_numpy()
    dorsal_t_test = []
    anterior_t_test = []
    for row_index in range(config.DATA_STEPS):
        dorsal_row_t_test = DataProcessing.t_test(da_np[row_index], dp_np[row_index])
        anterior_row_t_test = DataProcessing.t_test(aa_np[row_index], ap_np[row_index])
        dorsal_t_test.append(dorsal_row_t_test.pvalue >= 0.95)
        anterior_t_test.append(anterior_row_t_test.pvalue >= 0.95)

    dorsal_t_ntest = ~np.array(dorsal_t_test)
    anterior_t_ntest = ~np.array(anterior_t_test)

    # initialize graph, set Axes titles 
    fig, (axD, axA) = plt.subplots(2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    axD.title.set_text("Dorsal Axis")
    axA.title.set_text("Anterior Axis")

    dorsal_t_passed = list(itertools.compress(dorsal_t, dorsal_t_test))
    dorsal_t_npassed = list(itertools.compress(dorsal_t, dorsal_t_ntest))
    anterior_t_passed = list(itertools.compress(anterior_t, anterior_t_test))
    anterior_t_npassed = list(itertools.compress(anterior_t, anterior_t_ntest))

    da_average_passed = list(itertools.compress(da_average, dorsal_t_test))
    dp_average_passed = list(itertools.compress(dp_average, dorsal_t_test))
    da_average_npassed = list(itertools.compress(da_average, dorsal_t_ntest))
    dp_average_npassed = list(itertools.compress(dp_average, dorsal_t_ntest))

    aa_average_passed = list(itertools.compress(aa_average, anterior_t_test))
    ap_average_passed = list(itertools.compress(ap_average, anterior_t_test))
    aa_average_npassed = list(itertools.compress(aa_average, anterior_t_ntest))
    ap_average_npassed = list(itertools.compress(ap_average, anterior_t_ntest))

    # plot data
    axD.plot(dorsal_t, computed_dorsal_2, label="dorsal anterior model", markersize=2, color="blue")
    axD.plot(dorsal_t_passed, da_average_passed, "o", label="dorsal anterior data - passed", markersize=4, color="blue")
    axD.plot(dorsal_t_npassed, da_average_npassed, "o", label="dorsal anterior data", markersize=4, color='none', markeredgecolor="blue")

    axD.plot(dorsal_t, computed_dorsal_1, label="dorsal posterior model", markersize=2, color="red")
    axD.plot(dorsal_t_passed, dp_average_passed, "o", label="dorsal posterior data - passed", markersize=4, color="red")
    axD.plot(dorsal_t_npassed, dp_average_npassed, "o", label="dorsal posterior data", markersize=4, color='none', markeredgecolor="red")

    axA.plot(anterior_t, computed_anterior_2, label="anterior anterior model", markersize=2, color="blue")
    axA.plot(anterior_t_passed, aa_average_passed, "o", label="anterior anterior data - passed", markersize=4, color="blue")
    axA.plot(anterior_t_npassed, aa_average_npassed, "o", label="anterior anterior data", markersize=4, color='none', markeredgecolor="blue")

    axA.plot(anterior_t, computed_anterior_1, label="anterior posterior model", markersize=2, color="red")
    axA.plot(anterior_t_passed, ap_average_passed, "o", label="anterior posterior data - passed", markersize=4, color="red")
    axA.plot(anterior_t_npassed, ap_average_npassed, "o", label="anterior posterior data", markersize=4, color='none', markeredgecolor="red")


    # plot confidence bands
    axD.fill_between(dorsal_t, da_average - da_std, da_average + da_std, alpha=0.2, color="blue")
    axD.fill_between(dorsal_t, dp_average - dp_std, dp_average + dp_std, alpha=0.2, color="red")
    axA.fill_between(anterior_t, aa_average - aa_std, aa_average + aa_std, alpha=0.2, color="blue")
    axA.fill_between(anterior_t, ap_average - ap_std, ap_average + ap_std, alpha=0.2, color="red")

    # plot legend
    axD.legend()
    axA.legend()
    # save figure
    plt.savefig(config.PLOT_FIT)