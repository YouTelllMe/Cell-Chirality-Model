import pandas as pd
import numpy as np

def process_rawdf(df: pd.DataFrame, time_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assumes N=10. 
    """
    df = df.drop(index=[0]).reset_index(drop=True)
    time = df[time_name]
    df = df.drop(columns=[time_name])
    assert len(df.columns) == 20, "data format incorrect when processing raw df"
    df_right = df.iloc[:, 0:10]
    df_left = df.iloc[:, 10:20]

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
