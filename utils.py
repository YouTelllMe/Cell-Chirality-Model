import pandas as pd
import numpy as np
import config

def process_rawdf(df: pd.DataFrame, time_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    """
    df = df.drop(index=[0]).reset_index(drop=True)
    time = df[time_name]
    df = df.drop(columns=[time_name])
    assert len(df.columns) == config.DATA_N * 2, "data format incorrect when processing raw df"
    df_right = df.iloc[:, 0:config.DATA_N]
    df_left = df.iloc[:, config.DATA_N:config.DATA_N * 2]

    return (df_right, df_left, time)

    
def column_average(df: pd.DataFrame) -> np.ndarray:
    """
    return a column-wise average of a pandas dataframe
    """
    col_sum = 0
    num_cols = len(df.columns)
    for column in range(num_cols):
        col_sum += df.iloc[:,column].to_numpy()
    col_average = col_sum / num_cols
    return col_average


def get_data():
    """
    """
    # read xlsx files
    dorsal_xls = pd.ExcelFile(config.DORSAL_ANGLE_PATH)
    anterior_xls = pd.ExcelFile(config.ANTERIOR_ANGLE_PATH)
    # remove first two rows
    dorsal = pd.read_excel(dorsal_xls, "dorsal")
    anterior = pd.read_excel(anterior_xls, "anterior")
    dorsal_anterior, dorsal_posterior, dorsal_t = process_rawdf(dorsal, "Time(s)")
    anterior_anterior, anterior_dorsal, anterior_t = process_rawdf(anterior, "Time(s)")

    return(dorsal_anterior, dorsal_posterior, dorsal_t, 
           anterior_anterior, anterior_dorsal, anterior_t)