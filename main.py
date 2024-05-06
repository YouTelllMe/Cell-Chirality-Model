# from Fit.FitCellWall import fit_model_whole
# from Euler import Euler
# from Model.ModelABC import ModelABC
# from Modeling.Cell import FourCellSystem, Cell
# from Model.ModelCellWall import ModelCellWall
# from Model.ModelAB import ModelAB
# import config
# from Plot.PlotAll import plot_all
# from Least_Distance.minimize import find_min
# from Fit.FitCellWall import fit_model_whole
# import pandas as pd

# import config
# import pandas as pd


# if __name__ == "__main__":
#     # read xlsx files
#     dorsal_xls = pd.ExcelFile(config.DORSAL_ANGLE_PATH)
#     anterior_xls = pd.ExcelFile(config.ANTERIOR_ANGLE_PATH)
#     # remove first two rows
#     dorsal = pd.read_excel(dorsal_xls, "dorsal")
#     anterior = pd.read_excel(anterior_xls, "anterior")
#     ABa_dorsal, ABp_dorsal, dorsal_t = process_rawdf(dorsal, "Time(s)")
#     ABa_ant, ABp_ant, anterior_t = process_rawdf(anterior, "Time(s)")

#     processed_data = (ABa_dorsal, ABp_dorsal, ABa_ant, ABp_ant)



from modelling.Simulator import Simulator
from modelling.ModelAB import ModelAB
from plot.Animator import Animator
from utils import get_data
from modelling.Fit.FitCurveFit import fit_model_whole
from modelling.Fit.FitMinimize import fit_fmin_model

#TODO
"""
Fix Fit
Fix Plots
Fix Paths / Config
Fix Fitting of Cortical Flow


notes:
thing about the z plane and cells going beneath it
"""




sim = Simulator(ModelAB, (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5), A=24.391702590834782, B=0.019015984443137957, t_final=195)
# (24.391702590834782, 0.019015984443137957)
sim.run(True)
# SURFACES = [lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1]
# animator = Animator(sim.df)
# animator.animate()

# data = get_data()
# print(fit_fmin_model(get_data()))

animator = Animator(sim.df)
animator.animate()