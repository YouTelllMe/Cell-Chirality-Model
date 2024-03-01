import config
from matplotlib.pyplot import Axes
import pandas as pd


#TODO
def plot_xz(axX: Axes, axZ: Axes) -> None:
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
