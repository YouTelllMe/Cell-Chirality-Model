import utils
import config
from matplotlib.pyplot import Axes
import pandas as pd


def plot_angles(ax: Axes, 
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