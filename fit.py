import numpy as np
import pandas as pd 
import os
from models import model_AB
from euler_method import euler
import scipy
import matplotlib.pyplot as plt

def fit(x, anterior_anterior, anterior_dorsal, dorsal_anterior, dorsal_posterior):
    # initial point not included within tau
    t_final = 195
    A, B = x
    N = 400
    # takes 399 steps (initial position vector is directly added during euler)
    tau = np.linspace(1/N, 1, N-1)

    euler_data = euler(model_AB, 
                       np.array([-0.5,0.5,0.5, 0.5,0.5,0.5, -0.5,-0.5,0.5, 0.5,-0.5,0.5]), 
                       1/N, 
                       tau, 
                       A, 
                       B, 
                       t_final, 
                       False)
    
    computed_distances = euler_data[1]
    computed_angle = euler_data[2]

    computed_data_index = range(0, 400, 10)
    computed_dorsal_1 = computed_angle["dorsal1"][computed_data_index].to_numpy()
    computed_dorsal_2 = computed_angle["dorsal2"][computed_data_index].to_numpy()
    computed_anterior_1 = computed_angle["anterior1"][computed_data_index].to_numpy()
    computed_anterior_2 = computed_angle["anterior2"][computed_data_index].to_numpy()
    residual_square = 0
    for column_index in range(10):
        da_residual = dorsal_anterior.iloc[:,column_index].to_numpy() - computed_dorsal_2
        dp_residual = dorsal_posterior.iloc[:,column_index].to_numpy() - computed_dorsal_1
        aa_residual = anterior_anterior.iloc[:,column_index].to_numpy() - computed_anterior_2
        ad_residual = anterior_dorsal.iloc[:,column_index].to_numpy() - computed_anterior_1
        
        residual_square += (np.linalg.norm(da_residual) ** 2
                          + np.linalg.norm(dp_residual) ** 2
                          + np.linalg.norm(aa_residual) ** 2
                          + np.linalg.norm(ad_residual) ** 2)

    epsilon = 1
    residual_square += epsilon * ((np.sum(computed_distances["12"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["24"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["34"].to_numpy() - np.ones(400))) ** 2 +
                                (np.sum(computed_distances["13"].to_numpy() - np.ones(400))) ** 2)
    print(residual_square)
    return residual_square




