import os 
import matplotlib.pyplot as plt
import pandas as pd


CURR_DIR = os.getcwd()

data = sorted([file for file in os.listdir() if file[len(file)-4:] == ".csv"])
paths = [os.path.join(CURR_DIR,data_file) for data_file in data]


fig, (axX, axZ) = plt.subplots(1, 2)
axX.title.set_text("X Plot")
axZ.title.set_text("Z Plot")

t = None

COLORS = ["blue", "orange", "green", "red"]

for path_index in range(len(paths)):
    df = pd.read_csv(paths[path_index])
    if t is None:
        t = range(len(df))
    z = df["z"]
    x = df["x"]
    color = COLORS[path_index % 4]
    axX.plot(t, x, "-o", label=path_index+1, c = color, alpha = 0.4, markersize = 2)
    axZ.plot(t, z, "-o", label=path_index+1, c = color, alpha = 0.4, markersize = 2)

fig.set_figheight(7)
fig.set_figwidth(15)
axX.legend()
axZ.legend()
plt.savefig("xzplot.png")
    



