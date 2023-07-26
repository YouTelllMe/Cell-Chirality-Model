import os 
import matplotlib.pyplot as plt
import pandas as pd


CURR_DIR = os.getcwd()

data = ['data_1.csv', 'data_2.csv', 'data_3.csv', 'data_4.csv']
paths = [os.path.join(CURR_DIR,data_file) for data_file in data]
distances = pd.read_csv(os.path.join(CURR_DIR,"distance.csv"))
angles = pd.read_csv(os.path.join(CURR_DIR, "fit_angle.csv"))


fig, ((axX, axZ),(axDist, axDegree)) = plt.subplots(2, 2)
axX.title.set_text("X Plot")
axZ.title.set_text("Z Plot")
axDist.title.set_text("Distances")
axDegree.title.set_text("Theta vs Phi")

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



dorsal_xls = pd.ExcelFile(os.path.join(CURR_DIR, "dorsal.xlsx"))
anterior_xls = pd.ExcelFile(os.path.join(CURR_DIR, "anterior.xlsx"))
# remove first two rows
dorsal = pd.read_excel(dorsal_xls, "dorsal").drop(index=[0]).reset_index(drop=True)
anterior = pd.read_excel(anterior_xls, "anterior").drop(index=[0]).reset_index(drop=True)

# take time column
dorsal_t = dorsal["Time(s)"].to_numpy()
anterior_t = anterior["Time(s)"].to_numpy()
# drop time column
dorsal.drop(columns=["Time(s)"], inplace=True)
anterior.drop(columns=["Time(s)"], inplace=True)
assert len(dorsal.columns) == 20, "data format incorrect"

# dorsal_anterior is 24 (axis 2), dorsal_posterior is 13 (axis 1)
dorsal_anterior = dorsal.iloc[:,0:10]
dorsal_posterior = dorsal.iloc[:,10:20]
# anterior_anterior is 24 (axis 2), anterior_dorsal is 13 (axis 1)
anterior_anterior = anterior.iloc[:,0:10]
anterior_dorsal = anterior.iloc[:,10:20]

axDegree.plot(angles["dorsal2"].to_numpy(), angles["anterior2"].to_numpy(), "-o", markersize=2, c="orange")
for i in range(10):
    axDegree.plot(dorsal_anterior.iloc[:,i].to_numpy(), anterior_anterior.iloc[:,i].to_numpy(), "-o", markersize=2, c="black", alpha=0.4)


fig.set_figheight(7)
fig.set_figwidth(15)
axX.legend()
axZ.legend()
axDist.legend()
plt.savefig("xzplot.png")
    



