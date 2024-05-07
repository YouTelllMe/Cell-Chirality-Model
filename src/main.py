from modelling.Simulator import Simulator
from modelling.ModelAB import ModelAB
from modelling.ModelExtendingSpring import ModelExtendingSpring
from plot.Animator import Animator
from helpers import get_data, get_cortical_data
from modelling.Fit.FitCurveFit import fit_model_whole
from modelling.Fit.FitMinimize import fit_fmin_model
from modelling.Fit.FitCoritcalFlow import fit_cortical

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
    "FIT; SURFACES = [lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1]"
    print(fit_fmin_model(get_data()))
    # print(fit_model_whole(get_data()))


def run():
    "RUN, SAVE, AND ANIMATE"
    sim = Simulator(ModelAB, (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5), A=0.12868516070638591, B=1.0814748125814821, t_final=195,
                    surfaces=None)
    # A=0.12718164, B=0.06594284
    # A=0.12868516070638591, B=1.0814748125814821
    # sim = Simulator(ModelExtendingSpring, (0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, -0.5, 0, -0.5, 0.5, 0), 
    #                 A=0.01, B=1, t_final=195, surfaces=[lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1])
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
    # run()
    # plot_data()
    fit() 