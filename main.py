from Fit.FitCellWall import fit_model_whole
from Euler import Euler
from Model.ModelABC import ModelABC
from Cell import FourCellSystem, Cell
from Model.ModelCellWall import ModelCellWall
from Model.ModelAB import ModelAB
import config
from Plot.PlotAll import plot_all
from Least_Distance.minimize import find_min
from Fit.FitCurveFit import fit_model_wholeAB, fit_model_whole



# print(fit_model_whole())
# print(fit_model_wholeAB())
surface = lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 2

# model = ModelABC(1,1,1,FourCellSystem(
#                 Cell((-0.5,0.5,0.5)), 
#                 Cell((0.5,0.5,0.5)), 
#                 Cell((-0.5,-0.5,0.5)), 
#                 Cell((0.5,-0.5,0.5))))
modelAB = ModelAB(6.74025692, 0.03932766, 
                FourCellSystem(
                Cell((-0.5,0.5,0.5)), 
                Cell((0.5,0.5,0.5)), 
                Cell((-0.5,-0.5,0.5)), 
                Cell((0.5,-0.5,0.5)))
                  )

# modelWall = ModelCellWall(modelAB, surface, 0.1)

# Euler(modelAB).run(True)
plot_all()

# x0 = (0,0,0)
# surface = lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 1
# res = find_min(x0, surface)
# print(res.x)
