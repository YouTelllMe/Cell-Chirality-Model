import pandas as pd
import old.config as config
import matplotlib.pyplot as plt
from stats import resample_ci
from matplotlib.patches import Rectangle
from Fit.FitCurveFit import fit_model_whole




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