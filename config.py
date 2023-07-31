import os 


CUR_DIR = os.getcwd()

ANTERIOR_ANGLE_PATH = os.path.join(CUR_DIR, "raw_data", "anterior.xlsx")
CORTICALFLOW_PATH = os.path.join(CUR_DIR, "raw_data", "corticalflow.xlsx")
DORSAL_ANGLE_PATH = os.path.join(CUR_DIR, "raw_data", "dorsal.xlsx")

COLORS = ["blue", "orange", "green", "red"]
POSITION_1_DATAPATH = os.path.join(CUR_DIR, "data_1.csv")
POSITION_2_DATAPATH = os.path.join(CUR_DIR, "data_2.csv")
POSITION_3_DATAPATH = os.path.join(CUR_DIR, "data_3.csv")
POSITION_4_DATAPATH = os.path.join(CUR_DIR, "data_4.csv")

DISTANCE_DATAPATH = os.path.join(CUR_DIR, "distance.csv")
ANGLES_DATAPATH = os.path.join(CUR_DIR, "fit_angle.csv")
