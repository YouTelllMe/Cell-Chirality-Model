from Euler import EulerAnimator
import old.config as config
from matplotlib.pyplot import Axes
import pandas as pd


#TODO
def plot_xz(axX: Axes, axZ: Axes) -> None:
    """
    Plots the xz graph of the cell position-vectors on the given Axes.
    """

    position_df = pd.read_csv(config.POSITION_DATAPATH, index_col=0)
    cells_df = EulerAnimator.process_df(position_df)
    
    # plots x and z coordinates of the model position vectors 
    for path_index in range(len(cells_df)):
        df = cells_df[path_index]
        t = range(len(df))
        x = df["x"]
        z = df["z"]
        color = config.COLORS[path_index]
        axX.plot(t, x, "-o", label=path_index+1, c = color, markersize = 1)
        axZ.plot(t, z, "-o", label=path_index+1, c = color, markersize = 1)

    axX.legend()
    axZ.legend()
