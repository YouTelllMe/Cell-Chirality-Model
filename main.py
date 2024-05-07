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
SMT is messed up, check:
- equations on paper and in code
- angle calculations
- there should be a bug somewhere in the code that's causing these residuals to be wild



Fix Fit?
Fix Plots?


notes:
thing about the z plane and cells going beneath it
"""




sim = Simulator(ModelAB, (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5), A=4.683484906208419, B=1.2174858680302194, t_final=195)
# (4.683484906208419, 1.2174858680302194)
sim.run(True)
# SURFACES = [lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1]

# data = get_data()
# print(fit_model_whole(get_data()))

animator = Animator(sim.df)
animator.animate()