import numpy as np
from dataclasses import dataclass

class Cell:
    def __init__(self, position) -> None:
        self.position = np.array(position)

    def get_position(self):
        return self.position

@dataclass
class FourCellSystemDim:
    # TODO: better naming scheme
    dist_12: float
    dist_13: float
    dist_14: float
    dist_23: float
    dist_24: float
    dist_34: float
    ang_dorsal1: float
    ang_dorsal2: float
    ang_anterior1: float
    ang_anterior2: float
    


class FourCellSystem:

    def __init__(self, p1: Cell, p2: Cell, p3: Cell, p4: Cell) -> None:
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.info = self.get_dimensions()

    def get_dimensions(self) -> FourCellSystemDim:

        #position vectors
        position_1 = self.p1.get_position()
        position_2 = self.p2.get_position()
        position_3 = self.p3.get_position()
        position_4 = self.p4.get_position()

        # axis vectors
        axis_1 = position_1-position_3
        axis_2 = position_2-position_4
        anterior_ax_1 = np.array([axis_1[1], axis_1[2]])
        anterior_ax_2 = np.array([axis_2[1], axis_2[2]])
        dorsal_ax_1 = np.array([axis_1[0], axis_1[1]])
        dorsal_ax_2 = np.array([axis_2[0], axis_2[1]])
        # location of "0" degrees, used to dot with axis vectors to obtain angle
        dorsal_0 = np.array([1,0])
        anterior_0 = np.array([1,0])

        # dot product formula to obtain angle
        dorsal_ang_1 = np.arccos(np.dot(dorsal_ax_1, dorsal_0)/(np.linalg.norm(dorsal_ax_1)))
        dorsal_ang_2 = np.arccos(np.dot(dorsal_ax_2, dorsal_0)/(np.linalg.norm(dorsal_ax_2)))
        anterior_ang_1 = np.arccos(np.dot(anterior_ax_1, anterior_0)/(np.linalg.norm(anterior_ax_1)))
        anterior_ang_2 = np.arccos(np.dot(anterior_ax_2, anterior_0)/(np.linalg.norm(anterior_ax_2)))

        # checks quadrant of axis vector
        if dorsal_ax_1[1] < 0:
            dorsal_ang_1 *= -1
        if dorsal_ax_2[1] < 0:
            dorsal_ang_2 *= -1
        if anterior_ax_1[1] < 0:
            anterior_ang_1 *= -1
        if anterior_ax_2[1] < 0:
            anterior_ang_2 *= -1

        # compute Euclidean distance and conv angles to radians
        return FourCellSystemDim(np.linalg.norm(np.subtract(position_1, position_2)),
                                    np.linalg.norm(np.subtract(position_1, position_3)),
                                    np.linalg.norm(np.subtract(position_1, position_4)),
                                    np.linalg.norm(np.subtract(position_2, position_3)),
                                    np.linalg.norm(np.subtract(position_2, position_4)),
                                    np.linalg.norm(np.subtract(position_3, position_4)),
                                    dorsal_ang_1 * (180 / np.pi),
                                    dorsal_ang_2 * (180 / np.pi),
                                    anterior_ang_1 * (180 / np.pi),
                                    anterior_ang_2 * (180 / np.pi))
        
    def get_position(self) -> np.ndarray:
        return np.concatenate((self.p1.position, 
                        self.p2.position,
                        self.p3.position,
                        self.p4.position))

    

