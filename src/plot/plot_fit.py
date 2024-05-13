import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import scipy

def plot_fit(data, angles):
    """
    Plot model fit against raw data. 
    """
    (ABa_dorsal, ABp_dorsal, dorsal_t, ABa_ant, ABp_ant, anterior_t) = data
    
    da_average = column_average(ABa_dorsal)
    dp_average = column_average(ABp_dorsal)
    aa_average = column_average(ABa_ant)
    ap_average = column_average(ABp_ant)

    #TODO?
    # get standard error of the mean
    da_std = ABa_dorsal.std(axis=1).to_numpy() / np.sqrt(10)
    dp_std = ABp_dorsal.std(axis=1).to_numpy() / np.sqrt(10)
    aa_std = ABa_ant.std(axis=1).to_numpy() / np.sqrt(10)
    ap_std = ABp_ant.std(axis=1).to_numpy() / np.sqrt(10)

    # load data
    computed_ABa_dorsal = angles["ABa_dorsal"].to_numpy()
    computed_ABp_dorsal = angles["ABp_dorsal"].to_numpy()
    computed_ABa_ant = angles["ABa_anterior"].to_numpy()
    computed_ABp_ant = angles["ABp_anterior"].to_numpy()

    #t-test

    #each entry is a row
    da_np = ABa_dorsal.to_numpy()
    dp_np = ABp_dorsal.to_numpy()
    aa_np = ABa_ant.to_numpy()
    ap_np = ABp_ant.to_numpy()
   
    # boolean array; True if t test passes, False if not 
    dorsal_t_test = []
    anterior_t_test = []

    #check
    for row_index in range(40):
        # supression warning. All initial data are identical so can't perform t-test
        if (row_index != 0):
            dorsal_row_t_test = t_test(da_np[row_index], dp_np[row_index])
            anterior_row_t_test = t_test(aa_np[row_index], ap_np[row_index])
            # print(dorsal_row_t_test)
            # print(anterior_row_t_test)
            dorsal_t_test.append(dorsal_row_t_test.pvalue >= 0.95)
            anterior_t_test.append(anterior_row_t_test.pvalue >= 0.95)
        else:
            dorsal_t_test.append(True)
            anterior_t_test.append(True)

    # True if not pass, False if pass
    dorsal_t_ntest = ~np.array(dorsal_t_test)
    anterior_t_ntest = ~np.array(anterior_t_test)


    # initialize graph, set Axes titles 
    fig, (axD, axA) = plt.subplots(2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    axD.title.set_text("Dorsal View")
    axA.title.set_text("Anterior View")

    #TODO
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
    axD.plot(dorsal_t, computed_ABa_dorsal, label="ABa_dorsal model", markersize=2, color="blue")
    axD.plot(dorsal_t_passed, da_average_passed, "o", label="ABa_dorsal data - passed", markersize=4, color="blue")
    axD.plot(dorsal_t_npassed, da_average_npassed, "o", label="ABa_dorsal data", markersize=4, color='none', markeredgecolor="blue")

    axD.plot(dorsal_t, computed_ABp_dorsal, label="ABp_dorsal model", markersize=2, color="red")
    axD.plot(dorsal_t_passed, dp_average_passed, "o", label="ABp_dorsal data - passed", markersize=4, color="red")
    axD.plot(dorsal_t_npassed, dp_average_npassed, "o", label="ABp_dorsal data", markersize=4, color='none', markeredgecolor="red")

    axA.plot(anterior_t, computed_ABa_ant, label="ABa_ant model", markersize=2, color="blue")
    axA.plot(anterior_t_passed, aa_average_passed, "o", label="ABa_ant data - passed", markersize=4, color="blue")
    axA.plot(anterior_t_npassed, aa_average_npassed, "o", label="ABa_ant data", markersize=4, color='none', markeredgecolor="blue")

    axA.plot(anterior_t, computed_ABp_ant, label="ABp_ant model", markersize=2, color="red")
    axA.plot(anterior_t_passed, ap_average_passed, "o", label="ABp_ant data - passed", markersize=4, color="red")
    axA.plot(anterior_t_npassed, ap_average_npassed, "o", label="ABp_ant data", markersize=4, color='none', markeredgecolor="red")


    # plot confidence bands
    axD.fill_between(dorsal_t, da_average - da_std, da_average + da_std, alpha=0.2, color="blue")
    axD.fill_between(dorsal_t, dp_average - dp_std, dp_average + dp_std, alpha=0.2, color="red")
    axA.fill_between(anterior_t, aa_average - aa_std, aa_average + aa_std, alpha=0.2, color="blue")
    axA.fill_between(anterior_t, ap_average - ap_std, ap_average + ap_std, alpha=0.2, color="red")

    # plot legend
    axD.legend()
    axA.legend()
    # save figure
    plt.savefig('fit.png')


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


#TODO
def t_test(sample1: np.ndarray, sample2: np.ndarray):
    """
    Performs the t_test on two data samples for the null hypothesis that 2 independent samples
    have identical average (expected) values. 
    """
    t_test_result = scipy.stats.ttest_ind(sample1, sample2)
    return t_test_result
