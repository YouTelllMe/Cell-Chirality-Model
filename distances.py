import os 
import matplotlib.pyplot as plt
import pandas as pd


CURR_DIR = os.getcwd()

data = ['data_1.csv', 'data_2.csv', 'data_3.csv', 'data_4.csv']
paths = [os.path.join(CURR_DIR,data_file) for data_file in data]
distances = pd.read_csv(os.path.join(CURR_DIR,"distance.csv"))


fig, ((axX, axZ),(axDist, axDist2)) = plt.subplots(2, 2)
axX.title.set_text("X Plot")
axZ.title.set_text("Z Plot")
axDist.title.set_text("Distances")

t = None

COLORS = ["blue", "orange", "green", "red"]

for path_index in range(len(paths)):
    df = pd.read_csv(paths[path_index])
    if t is None:
        t = range(len(df))
    z = df["z"]
    x = df["x"]
    color = COLORS[path_index % 4]
    axX.plot(t, x, "-o", label=path_index+1, c = color, markersize = 1)
    axZ.plot(t, z, "-o", label=path_index+1, c = color, markersize = 1)

t = range(len(distances))
axDist.plot(t, distances["12"].to_numpy(), "-o", label="12", c="blue", markersize=1)
axDist.plot(t, distances["13"].to_numpy(), "-o", label="13", c="orange", markersize=1)
axDist.plot(t, distances["14"].to_numpy(), "-o", label="14", c="green", markersize=1)
axDist.plot(t, distances["23"].to_numpy(), "-o", label="23", c="red", markersize=1)
axDist.plot(t, distances["24"].to_numpy(), "-o", label="24", c="yellow", markersize=1)
axDist.plot(t, distances["34"].to_numpy(), "-o", label="34", c="pink", markersize=1)


fig.set_figheight(7)
fig.set_figwidth(15)
axX.legend()
axZ.legend()
axDist.legend()
plt.savefig("xzplot.png")
    



