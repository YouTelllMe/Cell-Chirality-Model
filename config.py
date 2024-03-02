import os 

#TODO (separate configs per module)
#===========================================================================#
"PATHS"
# curr dir
CUR_DIR = os.getcwd()
# model data
DISTANCE_DATAPATH = os.path.join(CUR_DIR,"model_output", "distance.csv")
ANGLES_DATAPATH = os.path.join(CUR_DIR,"model_output", "angle.csv")
POSITION_DATAPATH = os.path.join(CUR_DIR, "model_output", "position.csv")
RESAMPLE_DATAPATH = os.path.join(CUR_DIR, "model_output", "resample_AB.csv")
RESIDUAL_SQUARED_DATAPATH = os.path.join(CUR_DIR, "model_output", "residual_squared.csv")
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
PLOT_LEVEL_CURVE = os.path.join(CUR_DIR, "model_plot", "level_curve.png")
#===========================================================================#
"EULER + FIT"
T_FINAL = 195 # final timestamp of raw data
DATA_STEPS = 40
DATA_N = 10 
STEP_SCALE = 10 
MODEL_STEPS = DATA_STEPS * STEP_SCALE # how many ODE steps per data step
h = 1/MODEL_STEPS # total time = 1 unit over number of step size
GUESSABC = (1, 1, 1) #[9.52396779, 0.0642205, 5.767196]
GUESSAB = (1, 1) #[6.83606077, 0.03953272]
WALL = (0)
#===========================================================================#
"OTHER"
COLORS = ["blue", "orange", "green", "red"] # colors for cells
LEVEL_CURVE_BINS = [154000, 156000, 158000, 160000, 170000, 180000, 190000, 200000, 250000, 300000, 350000, 400000]





