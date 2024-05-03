import DataProcessing
from models import model_AB
import numpy as np
from euler_method import euler
import config
from collections.abc import Sequence
import pandas as pd
import random 
from Fit.FitModel import fit_model_whole, fit
from collections.abc import Sequence


def resample_ci(CI: int = 0.95,
                resampled: tuple[Sequence, Sequence] | None = None):
    """
    CI must be between 0 and 1 (if larger or smaller, casted to 0 and 1)
    """
    CI = min(CI, 1)
    CI = max(CI, 0)

    if resampled is None:
        resample_df = pd.read_csv(config.RESAMPLE_DATAPATH)
        A_n = resample_df["A"].to_numpy()
        B_n = resample_df["B"].to_numpy()
    else:
        A_n, B_n = resampled
        A_n = np.array(A_n)
        B_n = np.array(B_n)

    n_folds = len(A_n)
    CI_side = int((1 - CI) / 2 * n_folds)
    A_n_sorted = np.sort(A_n)
    B_n_sorted = np.sort(B_n)

    # indexing a bit tricky here
    A_CI = [A_n_sorted[CI_side], A_n_sorted[-CI_side-1]]
    B_CI = [B_n_sorted[CI_side], B_n_sorted[-CI_side-1]]

    return(A_CI, B_CI)


def resample_n(n_folds: int = 10,
               save: bool = False):
    """
    """
    original_data = DataProcessing.get_data()

    A, B = fit_model_whole()[0]
    res = residuals(original_data, A, B)

    A_n = []
    B_n = []

    for fold_index in range(n_folds):
        resampled = resample(original_data, res)
        resampled_A, resampled_B = fit_model_whole(resampled)[0]
        print(f"fold {fold_index}: {resampled_A}, {resampled_B}")
        A_n.append(resampled_A)
        B_n.append(resampled_B)
    
    if save:
        resample_df = pd.DataFrame(data={"A":A_n, "B":B_n})
        resample_df.to_csv(config.RESAMPLE_DATAPATH)
    return(A_n, B_n)


def resample(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame, pd.DataFrame], residuals: dict[str, set[float]]):
    """"""
    (dorsal_anterior, 
     dorsal_posterior, 
     dorsal_t,
     anterior_anterior, 
     anterior_dorsal, 
     anterior_t) = data
    
    zero_matrix = np.zeros((config.DATA_STEPS, config.DATA_N))
    resample = {"dorsal1": np.copy(zero_matrix), 
                "dorsal2": np.copy(zero_matrix), 
                "anterior1": np.copy(zero_matrix), 
                "anterior2": np.copy(zero_matrix)}

    for angle_type in list(residuals.keys()):
        sample = residuals[angle_type]
        for row_index in range(config.DATA_STEPS):
            for column_index in range(config.DATA_N):
                # generate random int
                random_index = random.randint(0, (config.DATA_STEPS * config.DATA_N) - 1)
                resample[angle_type][row_index][column_index] = sample[random_index]

    resample["dorsal2"] = resample["dorsal2"] + dorsal_anterior.to_numpy() 
    resample["dorsal1"] = resample["dorsal1"] + dorsal_posterior.to_numpy()
    resample["anterior2"] = resample["anterior2"] + anterior_anterior.to_numpy()
    resample["anterior1"] = resample["anterior1"] + anterior_dorsal.to_numpy()

    # change later; is there a need for it to be pd dataframes? or can it just be a numpy matrix
    return (pd.DataFrame(data=resample["dorsal2"], columns=dorsal_anterior.columns),
            pd.DataFrame(data=resample["dorsal1"], columns=dorsal_posterior.columns),
            dorsal_t,
            pd.DataFrame(data=resample["anterior2"], columns=anterior_anterior.columns),
            pd.DataFrame(data=resample["anterior1"], columns=anterior_dorsal.columns),
            anterior_t,
            )
    

def get_std(df: pd.DataFrame, axis: int = 0):
    """
    Return the standard deviation across the specified axis.
    """
    return df.std(axis=axis).to_numpy()


def t_test(sample1: np.ndarray, sample2: np.ndarray):
    """
    Performs the t_test on two data samples for the null hypothesis that 2 independent samples
    have identical average (expected) values. 
    """
    t_test_result = scipy.stats.ttest_ind(sample1, sample2)
    return t_test_result
