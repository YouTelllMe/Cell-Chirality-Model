import os 
import matplotlib.pyplot as plt
import pandas as pd
import config
import utils
from matplotlib.axes import Axes
from matplotlib.figure import Figure


position_datapaths = [config.POSITION_1_DATAPATH,
                      config.POSITION_2_DATAPATH,
                      config.POSITION_3_DATAPATH,
                      config.POSITION_4_DATAPATH]

distances = pd.read_csv(config.DISTANCE_DATAPATH)
angles = pd.read_csv(config.ANGLES_DATAPATH)


fig, ((axX, axZ),(axDist, axDegree)) = plt.subplots(2, 2)
axX.title.set_text("X Plot")
axZ.title.set_text("Z Plot")
axDist.title.set_text("Distances")
axDegree.title.set_text("Theta vs Phi")

for path_index in range(len(position_datapaths)):

    df = pd.read_csv(position_datapaths[path_index])

    color = config.COLORS[path_index % 4]
    t = range(len(df))
    x = df["x"]
    z = df["z"]

    axX.plot(t, x, "-o", label=path_index+1, c = color, markersize = 1)
    axZ.plot(t, z, "-o", label=path_index+1, c = color, markersize = 1)


t = range(len(distances))
axDist.plot(t, distances["12"].to_numpy(), "-o", label="12", c="blue", markersize=1)
axDist.plot(t, distances["13"].to_numpy(), "-o", label="13", c="orange", markersize=1)
axDist.plot(t, distances["14"].to_numpy(), "-o", label="14", c="green", markersize=1)
axDist.plot(t, distances["23"].to_numpy(), "-o", label="23", c="red", markersize=1)
axDist.plot(t, distances["24"].to_numpy(), "-o", label="24", c="yellow", markersize=1)
axDist.plot(t, distances["34"].to_numpy(), "-o", label="34", c="pink", markersize=1)


# remove first two rows
anterior_xls = pd.ExcelFile(config.ANTERIOR_ANGLE_PATH)
dorsal_xls = pd.ExcelFile(config.DORSAL_ANGLE_PATH)

anterior = pd.read_excel(anterior_xls, "anterior")
anterior_anterior, anterior_dorsal, anterior_t = utils.process_rawdf(anterior, "Time(s)")

dorsal = pd.read_excel(dorsal_xls, "dorsal")
dorsal_anterior, dorsal_posterior, dorsal_t = utils.process_rawdf(dorsal, "Time(s)")

axDegree.plot(angles["dorsal2"].to_numpy(), 
              angles["anterior2"].to_numpy(), 
              "-o", markersize=2, 
              c="orange")
for i in range(10):
    axDegree.plot(dorsal_anterior.iloc[:,i].to_numpy(), 
                  anterior_anterior.iloc[:,i].to_numpy(), 
                  "-o", 
                  markersize=2, 
                  c="black", 
                  alpha=0.4)
    
fig.set_figheight(7)
fig.set_figwidth(15)
axX.legend()
axZ.legend()
axDist.legend()
plt.savefig("xzplot.png")
    



def plot_distance(Ax: Axes) -> None:
    pass

def plot_xz_plot(AXx: Axes | None, Axz: Axes | None) -> None:
    pass

def plot_thetaphi(Ax: Axes) -> None:
    pass

def plot_all() -> None:
    pass