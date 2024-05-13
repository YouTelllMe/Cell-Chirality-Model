from modelling.Simulator import Simulator
from modelling.models import ModelAB
from modelling.models import ModelExtendingSpring
from plot.Animator import Animator
from helpers import get_data, get_cortical_data
from modelling.fit.FitCurveFit import fit_model_whole
from modelling.fit.FitMinimize import fit_fmin_model
from modelling.fit.FitCoritcalFlow import fit_cortical

import matplotlib.pyplot as plt

from plot.plot_angles import plot_angles
from plot.plot_distances import plot_distance
from plot.plot_xz import plot_xz
from plot.plot_fit import plot_fit

import os
import pandas as pd

#TODO
"""
SMT is messed up, check:
- equations on paper and in code
- angle calculations
- there should be a bug somewhere in the code that's causing these residuals to be wild
    - investigate by looking at the force components
    - dorsal is kinda weird 
    - there's 100% a shit that's wrong with the string equations


Fix Fit?
Fix Plots?


notes:
- thing about the z plane and cells going beneath it
- what was wrong with the standard error of the mean in the plot again?
"""

def fit():
    "FIT; make sure to customize the model in the fit functions"
    # print(fit_fmin_model(get_data()))
    print(fit_model_whole(get_data()))



def run():
    "RUN, SAVE, AND ANIMATE"
    sim = Simulator(ModelAB.get_velocity(0.12693928, 0.06383564, 195), (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5))
    # sim = Simulator(ModelExtendingSpring.get_velocity(0.12693815, 0.06383225, 195), (0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, -0.5, 0, -0.5, 0.5, 0))
    sim.run(True)
    animator = Animator(sim.df)
    animator.animate()




def plot_data():
    "PLOTTING"
    distances_xls = pd.ExcelFile("distances.xlsx")
    angles_xls = pd.ExcelFile("angles.xlsx")
    output_xls = pd.ExcelFile("output.xlsx")
    distances = pd.read_excel(distances_xls, "Sheet1")
    angles = pd.read_excel(angles_xls, "Sheet1")
    output = pd.read_excel(output_xls, "Sheet1")


    data = get_data()
    (ABa_dorsal, ABp_dorsal, dorsal_t, ABa_ant, ABp_ant, anterior_t) = data

    fig, ((axX, axZ),(axDist, axDegree)) = plt.subplots(2, 2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    axX.title.set_text("X Plot")
    axZ.title.set_text("Z Plot")
    axDist.title.set_text("Distances")
    axDegree.title.set_text("Theta vs Phi")

    # # run plotting helper functions; saves figure
    plot_distance(axDist, distances)
    plot_angles(axDegree, angles, ABa_dorsal, ABp_dorsal, ABa_ant, ABp_ant)
    plot_xz(axX, axZ, output)
    plt.savefig('xzpng')

    plot_fit(data, angles)



def fit_cortical_flow():
    "FIT CORTICAL FLOW"
    data = get_cortical_data()
    fit_cortical(data)


if __name__ == "__main__":
    run()
    plot_data()
    # fit() 