import os 
import matplotlib.pyplot as plt
import pandas as pd
import config
import itertools
import utils
import numpy as np 
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from fit import fit_model_whole
from stats import resample_ci



def plot_all() -> None:
    """
    Plots the distance, xz, thetaphi, fit plots from the model_output data folder. 
    """

    # read distance data
    distances = pd.read_csv(config.DISTANCE_DATAPATH)
    # read angles data
    angles = pd.read_csv(config.ANGLES_DATAPATH)
    anterior_xls = pd.ExcelFile(config.ANTERIOR_ANGLE_PATH)
    dorsal_xls = pd.ExcelFile(config.DORSAL_ANGLE_PATH)
    anterior = pd.read_excel(anterior_xls, "anterior")
    dorsal = pd.read_excel(dorsal_xls, "dorsal")

    # initialize figure and axes, set configs
    fig, ((axX, axZ),(axDist, axDegree)) = plt.subplots(2, 2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    axX.title.set_text("X Plot")
    axZ.title.set_text("Z Plot")
    axDist.title.set_text("Distances")
    axDegree.title.set_text("Theta vs Phi")

    # run plotting helper functions; saves figure
    plot_distance(axDist, distances)
    plot_thetaphi(axDegree, angles, anterior, dorsal)
    plot_xz_plot(axX, axZ)
    plt.savefig(config.PLOT_xzplot)

    plot_fit()

   
def plot_distance(ax: Axes, data: pd.DataFrame) -> None:
    """
    Plots the inter-cell distances given the data on the given Axes. 
    """
    t = range(len(data))
    ax.plot(t, data["12"].to_numpy(), "-o", label="12", c="blue", markersize=1)
    ax.plot(t, data["13"].to_numpy(), "-o", label="13", c="orange", markersize=1)
    ax.plot(t, data["14"].to_numpy(), "-o", label="14", c="green", markersize=1)
    ax.plot(t, data["23"].to_numpy(), "-o", label="23", c="red", markersize=1)
    ax.plot(t, data["24"].to_numpy(), "-o", label="24", c="yellow", markersize=1)
    ax.plot(t, data["34"].to_numpy(), "-o", label="34", c="pink", markersize=1)
    ax.legend()


def plot_xz_plot(axX: Axes, axZ: Axes) -> None:
    """
    Plots the xz graph of the cell position-vectors on the given Axes.
    """
    position_datapaths = [config.POSITION_1_DATAPATH,
                        config.POSITION_2_DATAPATH,
                        config.POSITION_3_DATAPATH,
                        config.POSITION_4_DATAPATH]
    
    # plots x and z coordinates of the model position vectors 
    for path_index in range(len(position_datapaths)):
        df = pd.read_csv(position_datapaths[path_index])
        t = range(len(df))
        x = df["x"]
        z = df["z"]
        color = config.COLORS[path_index % 4]
        axX.plot(t, x, "-o", label=path_index+1, c = color, markersize = 1)
        axZ.plot(t, z, "-o", label=path_index+1, c = color, markersize = 1)

    axX.legend()
    axZ.legend()


def plot_thetaphi(ax: Axes, 
                  angles_df: pd.DataFrame, 
                  anterior_df: pd.DataFrame, 
                  dorsal_df: pd.DataFrame
                  ) -> None:
    """
    Plots the Theta (anterior) vs Phi (dorsal) graph on the given Axes.
    """
    # process raw data
    anterior_anterior, _, _ = utils.process_rawdf(anterior_df, "Time(s)")
    dorsal_anterior, _, _ = utils.process_rawdf(dorsal_df, "Time(s)")

    # plots model theta vs phi
    ax.plot(angles_df["dorsal2"].to_numpy(), 
                angles_df["anterior2"].to_numpy(), 
                "-o", 
                markersize=2, 
                c="orange")
    
    # plot data theta vs phi 
    for i in range(config.DATA_N):
        ax.plot(dorsal_anterior.iloc[:,i].to_numpy(), 
                anterior_anterior.iloc[:,i].to_numpy(), 
                "-o", 
                markersize=2, 
                c="black", 
                alpha=0.4)


def plot_fit():
    """
    Plot model fit against raw data. 
    """
    (dorsal_anterior, 
     dorsal_posterior, 
     dorsal_t,
     anterior_anterior, 
     anterior_posterior, 
     anterior_t) = utils.get_data()
    
    # average angles for plotting
    da_average = utils.column_average(dorsal_anterior)
    dp_average = utils.column_average(dorsal_posterior)
    aa_average = utils.column_average(anterior_anterior)
    ap_average = utils.column_average(anterior_posterior)

    # get standard error of the mean
    da_std = utils.get_std(dorsal_anterior, axis=1) / np.sqrt(config.DATA_STEPS)
    dp_std = utils.get_std(dorsal_posterior, axis=1) / np.sqrt(config.DATA_STEPS)
    aa_std = utils.get_std(anterior_anterior, axis=1) / np.sqrt(config.DATA_STEPS)
    ap_std = utils.get_std(anterior_posterior, axis=1) / np.sqrt(config.DATA_STEPS)

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
        dorsal_row_t_test = utils.t_test(da_np[row_index], dp_np[row_index])
        anterior_row_t_test = utils.t_test(aa_np[row_index], ap_np[row_index])
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


def plot_level_curves() -> None:
    """
    """

    residual_squared_df = pd.read_csv(config.RESIDUAL_SQUARED_DATAPATH)
    A_all = residual_squared_df["A"].to_numpy()
    B_all = residual_squared_df["B"].to_numpy()
    residual_squared_df.drop(columns=["A", "B"], inplace=True)
    residual_squared = residual_squared_df.to_numpy()

    fig1, ax = plt.subplots()

    colors = []
    color_interval = 1 / len(config.LEVEL_CURVE_BINS)
    for color_index in range(len(config.LEVEL_CURVE_BINS)):
        color_val = color_index * color_interval
        colors.append((color_val, color_val, color_val))

    contour = ax.contourf(B_all, 
                          A_all, 
                          residual_squared, 
                          config.LEVEL_CURVE_BINS,
                          colors = colors)
    resample_CI = resample_ci(0.95)
    resample_patch = Rectangle((resample_CI[1][0], resample_CI[0][0]), 
                                   resample_CI[1][1]-resample_CI[1][0], 
                                   resample_CI[0][1]-resample_CI[0][0], 
                                   linewidth=1,
                                   edgecolor="r",
                                   facecolor="none",
                                   label="resample confidence")
    AB, cov_matrix, AB_uncertainty = fit_model_whole()
    model_fit_patch = Rectangle((AB[1] - AB_uncertainty[1], AB[0] - AB_uncertainty[0]), 
                                   AB_uncertainty[1] * 2, 
                                   AB_uncertainty[0] * 2, 
                                   linewidth=1,
                                   edgecolor="g",
                                   facecolor="none",
                                   label="curve fit confidence")
    
    ax.add_patch(resample_patch)
    ax.add_patch(model_fit_patch)
    ax.plot(AB[1], AB[0], "o", color = "g", markersize=5, label="curve fit best")
    ax.annotate(f"({round(AB[1], 3)},{round(AB[0], 2)})", xy=(AB[1], AB[0]), color = "g")
    ax.legend()
    fig1.colorbar(contour)
    plt.savefig(config.PLOT_LEVEL_CURVE)
    plt.show()

if __name__ == "__main__":
    plot_all()