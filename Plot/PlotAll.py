import pandas as pd
import config
import matplotlib.pyplot as plt
from Plot.PlotDistance import plot_distance
from Plot.PlotAngles import plot_angles
from Plot.PlotXZ import plot_xz
from Plot.PlotFit import plot_fit 


def plot_all() -> None:
    """
    Plots the distance, xz, thetaphi, fit plots from the model_output data folder. 
    """

    # read distance data
    distances = pd.read_csv(config.DISTANCE_DATAPATH, index_col=0)
    # read angles data
    angles = pd.read_csv(config.ANGLES_DATAPATH, index_col=0)
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
    plot_angles(axDegree, angles, anterior, dorsal)
    plot_xz(axX, axZ)
    plt.savefig(config.PLOT_xzplot)

    plot_fit()

