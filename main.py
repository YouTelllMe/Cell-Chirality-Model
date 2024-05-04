from Fit.FitCellWall import fit_model_whole
from Euler import Euler
from Model.ModelABC import ModelABC
from Modeling.Cell import FourCellSystem, Cell
from Model.ModelCellWall import ModelCellWall
from Model.ModelAB import ModelAB
import config
from Plot.PlotAll import plot_all
from Least_Distance.minimize import find_min
from Fit.FitCellWall import fit_model_whole
import pandas as pd

def process_rawdf(df: pd.DataFrame, time_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process Raw Excel Data into 
    """
    df = df.drop(index=[0]).reset_index(drop=True)
    time = df[time_name]
    df = df.drop(columns=[time_name])
    assert len(df.columns) == config.DATA_N * 2, "data format incorrect when processing raw df"
    df_right = df.iloc[:, 0:config.DATA_N]
    df_left = df.iloc[:, config.DATA_N:config.DATA_N * 2]

    return (df_right, df_left, time)


if __name__ == "__main__":
    # read xlsx files
    dorsal_xls = pd.ExcelFile(config.DORSAL_ANGLE_PATH)
    anterior_xls = pd.ExcelFile(config.ANTERIOR_ANGLE_PATH)
    # remove first two rows
    dorsal = pd.read_excel(dorsal_xls, "dorsal")
    anterior = pd.read_excel(anterior_xls, "anterior")
    dorsal_anterior, dorsal_posterior, dorsal_t = process_rawdf(dorsal, "Time(s)")
    anterior_anterior, anterior_posterior, anterior_t = process_rawdf(anterior, "Time(s)")

    processed_data = (dorsal_anterior, dorsal_posterior, dorsal_t, 
            anterior_anterior, anterior_posterior, anterior_t)


