from Cell import FourCellSystem, Cell
from Model.ModelABC import ModelABC
from Model.Model import Model
from Least_Distance.minimize import find_min
import numpy as np

class ModelCellWall():
     def __init__(self, model: Model, surface, push_factor) -> None:
          self.model = model
          self.system = model.system
          self.surface = surface
          self.push_factor = push_factor
    
     def step(self, time):
          curr_model = self.model.step(time)  
          curr_model.system.p1 = self.update_cell(curr_model.system.p1)
          curr_model.system.p2 = self.update_cell(curr_model.system.p2)
          curr_model.system.p3 = self.update_cell(curr_model.system.p3)
          curr_model.system.p4 = self.update_cell(curr_model.system.p4)

          return ModelCellWall(
                    curr_model,
                    self.surface, 
                    self.push_factor)

     def update_cell(self, cell: Cell) -> Cell:
          position = cell.get_position()
          min_point = find_min(position, self.surface)
          norm = np.linalg.norm(min_point.x-position)
          if norm < 0.5:  
               oldposition = position
               position = position + self.push_factor * (0.5 - norm) * (min_point.x-position)/norm
               print(oldposition, "->", position)
          return Cell(position)








        

    