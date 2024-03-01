from Fit.FitCellWall import fit_model_whole
from Euler import Euler
from Model.ModelABC import ModelABC
from Cell import FourCellSystem, Cell
from Model.ModelCellWall import ModelCellWall
import config
from Plot.PlotAll import plot_all
from Least_Distance.minimize import find_min



# print(fit_model_whole())
surface = lambda x: x[0]**2 + (x[1]**2)/3 + x[2]**2 - 1

model = ModelABC(1,1,1,FourCellSystem(
                Cell((-0.5,0.5,0.5)), 
                Cell((0.5,0.5,0.5)), 
                Cell((-0.5,-0.5,0.5)), 
                Cell((0.5,-0.5,0.5))))
Euler(ModelCellWall(model, surface, 0.0001)).run(True)

# plot_all()

# x0 = (0,0,0)
# surface = lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 1
# res = find_min(x0, surface)
# print(res.x)
