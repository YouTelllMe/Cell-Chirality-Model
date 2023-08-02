import utils
from models import model_AB
import numpy as np
from euler_method import euler
import config
from collections.abc import Sequence
import pandas as pd
import random 
from fit import fit_model_whole


def resample_n(n_folds: int = 10):
    """
    """
    original_data = utils.get_data()

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
    
            
def residuals(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame, pd.DataFrame], A: float, B: float):
    """
    """

    (dorsal_anterior, 
     dorsal_posterior, 
     dorsal_t,
     anterior_anterior, 
     anterior_dorsal, 
     anterior_t) = data

    residuals = {"dorsal1": [], "dorsal2": [], "anterior1": [], "anterior2": []}
    
    # takes 400 steps (initial position vector inclusive)
    N = config.MODEL_STEPS
    tau = np.linspace(1/N, 1, N-1)
    euler_data = euler(model_AB, 
                        np.array([-0.5,0.5,0.5, 0.5,0.5,0.5, -0.5,-0.5,0.5, 0.5,-0.5,0.5]), 
                        1/N, 
                        tau, 
                        A, 
                        B, 
                        False)
        
    computed_angle = euler_data[2]
    computed_data_index = range(0, config.MODEL_STEPS, config.STEP_SCALE)
    computed_dorsal_1 = computed_angle["dorsal1"][computed_data_index].to_numpy()
    computed_dorsal_2 = computed_angle["dorsal2"][computed_data_index].to_numpy()
    computed_anterior_1 = computed_angle["anterior1"][computed_data_index].to_numpy()
    computed_anterior_2 = computed_angle["anterior2"][computed_data_index].to_numpy()

    for column_index in range(config.DATA_N):
        residuals["dorsal1"].extend(dorsal_posterior.iloc[:,column_index].to_numpy() - computed_dorsal_1)
        residuals["dorsal2"].extend(dorsal_anterior.iloc[:,column_index].to_numpy() - computed_dorsal_2)
        residuals["anterior1"].extend(anterior_anterior.iloc[:,column_index].to_numpy() - computed_anterior_2)
        residuals["anterior2"].extend(anterior_dorsal.iloc[:,column_index].to_numpy() - computed_anterior_1)

    return residuals


if __name__ == "__main__":
    res = residuals(utils.get_data(), 6.7, 0.04)
    resample(utils.get_data(),res)