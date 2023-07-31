import os 





#===========================================================================#
"PATHS"
# curr dir
CUR_DIR = os.getcwd()
# model data
DISTANCE_DATAPATH = os.path.join(CUR_DIR,"model_output", "distance.csv")
ANGLES_DATAPATH = os.path.join(CUR_DIR,"model_output", "fit_angle.csv")
POSITION_1_DATAPATH = os.path.join(CUR_DIR, "model_output", "data_1.csv")
POSITION_2_DATAPATH = os.path.join(CUR_DIR, "model_output", "data_2.csv")
POSITION_3_DATAPATH = os.path.join(CUR_DIR, "model_output", "data_3.csv")
POSITION_4_DATAPATH = os.path.join(CUR_DIR, "model_output", "data_4.csv")
# raw data
ANTERIOR_ANGLE_PATH = os.path.join(CUR_DIR, "raw_data", "anterior.xlsx")
CORTICALFLOW_PATH = os.path.join(CUR_DIR, "raw_data", "corticalflow.xlsx")
DORSAL_ANGLE_PATH = os.path.join(CUR_DIR, "raw_data", "dorsal.xlsx")
#plot output
PLOT_FIT = os.path.join(CUR_DIR, "model_plot", "fit.png")
PLOT_XY = os.path.join(CUR_DIR, "model_plot", "XY.png")
PLOT_XYZ = os.path.join(CUR_DIR, "model_plot", "XYZ.png")
PLOT_XZ = os.path.join(CUR_DIR, "model_plot", "XZ.png")
PLOT_YZ = os.path.join(CUR_DIR, "model_plot", "YZ.png")
PLOT_xzplot = os.path.join(CUR_DIR, "model_plot", "xzplot.png")
PLOT_FIT_CORTICAL = os.path.join(CUR_DIR, "model_plot", "fit_cortical.png")
#===========================================================================#
"EULER + FIT"
T_FINAL = 195 # final timestamp of raw data
DATA_N = 10 # 
MODEL_STEPS = 400 
GUESS = []
#===========================================================================#
"OTHER"
COLORS = ["blue", "orange", "green", "red"] # colors for cells





