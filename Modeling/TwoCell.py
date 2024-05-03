import os
from Cell import Cell
import math
from Euler import EulerAnimator
from Least_Distance.minimize import find_min
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import config

class TwoCell:

    def __init__(self, cur_pos, spring_constant, a) -> None:
        self.a = a
        self.k = spring_constant
        self.cur_pos = np.array(cur_pos)
        self.euler = [cur_pos]


        self.STEPS = 500
        self.H = 1/self.STEPS # 1/500; why do we do h = 1/ steps again? 
        # self.FORCE_T = lambda t: 10*t*np.e**-(5*t)
        # self.FORCE_T = lambda t: 0.5
        self.SURFACES = [lambda x: x[1] - 1, lambda x: x[1] + 1]
        

    def get_velocity(self, t):
        cell1 = np.array([self.cur_pos[0], self.cur_pos[1], self.cur_pos[2]])
        cell2 = np.array([self.cur_pos[3], self.cur_pos[4], self.cur_pos[5]])
        
        distance = np.linalg.norm(cell2-cell1)

        u12 = (cell1-cell2) / distance # unit vector from 2 to 1
        
        # 0.6449 is the avg lambda of left right cortical flow 
        # growing spring force, velocity = spring force? also, calculation here correct? Shouldn't it be rest - displacement? 
        # 1 is initial spring rest length
        # should I multiply by a tfinal? if we normalized the time? increase h? 
        springforce = - self.k * (distance - (1 + self.a *(0.6449**2 - (0.6449*t + 0.6449**2)*np.e**(-t/0.6449))))
        cell1_prime = u12 * springforce
        cell2_prime = -u12 * springforce

        # cell wall forces 
        for surface in self.SURFACES:
            cell1_prime += self.cell_wall_step(cell1, surface)
            cell2_prime += self.cell_wall_step(cell2, surface)

        return np.concatenate((cell1_prime, cell2_prime))
    
    def step(self, t):
        self.cur_pos = self.cur_pos + self.get_velocity(t) * self.H
        self.euler.append(self.cur_pos)

    def run(self):
        time = np.linspace(1/self.STEPS, 1, self.STEPS-1)
        for t in time:
            self.step(t)

        self.df = pd.DataFrame(self.euler)
        self.animate()

    def cell_wall_step(self, pos, surface):
        min_point = find_min(pos, surface)
        norm = np.linalg.norm(min_point.x-pos)
        if norm < 0.5:  
            print(np.array(self.k * (0.5 - norm) * (pos-min_point.x)/norm))
            return np.array(self.k * (0.5 - norm) * (pos-min_point.x)/norm)
        return np.zeros(len(pos))
    
    def animate(self):
        # setup figure and axes
        FIG = plt.figure()
        AX = FIG.add_subplot(projection="3d")
        # setting range 
        SIZE = 2
        AX.set(xlim3d=(-SIZE, SIZE), xlabel='X')
        AX.set(ylim3d=(-SIZE, SIZE), ylabel='Y')
        AX.set(zlim3d=(-SIZE, SIZE), zlabel='Z')


        cell_dfs = EulerAnimator.process_df(self.df)
        cell_dfs[0].to_csv(os.path.join(os.getcwd(), "cell1.csv"))
        cell_dfs[1].to_csv(os.path.join(os.getcwd(), "cell2.csv"))
        
        positions_vectors = []
        animated_indicies = range(0, self.STEPS-1, 10)
        for position_index in range(len(cell_dfs)):
            new_position, = AX.plot([],[],[],".", alpha=0.4, markersize=3, label=position_index+1)
            positions_vectors.append(new_position)
            cell_dfs[position_index] = cell_dfs[position_index].iloc[animated_indicies].reset_index(drop=True)

        steps = len(cell_dfs[0].index)
        AX.legend()


        # animate, run update_replace at every step
        anim = FuncAnimation(FIG, 
                            self.update_replace, 
                            steps, 
                            fargs=(cell_dfs,
                                SIZE,
                                positions_vectors, 
                                ),
                            interval=1,
                            repeat=True,
                            )
        
        # save animation from different angles
        anim.save(os.path.join(os.getcwd(), "2-cell", "2cell-1.png"))
        AX.view_init(90, -90, 0)
        anim.save(os.path.join(os.getcwd(), "2-cell", "2cell-2.png"))
        AX.view_init(0, -90, 0)
        anim.save(os.path.join(os.getcwd(), "2-cell", "2cell-3.png"))
        AX.view_init(0, 0, 0)
        anim.save(os.path.join(os.getcwd(), "2-cell", "2cell-4.png"))
        

    def update_replace(self, 
                       frame, 
                    data, 
                    SIZE, 
                    positions,):
        
        # update position vector, draw spheres
        for curve_index in range(len(positions)):
            x = data[curve_index]['x'][frame]
            y = data[curve_index]['y'][frame]
            z = data[curve_index]['z'][frame]
            ball = self.generate_ball([x,y,z], 0.5)
            ball["x"].append(x)
            ball["y"].append(y)
            ball["z"].append(z)
            positions[curve_index].set_data([ball["x"], ball["y"]])
            positions[curve_index].set_3d_properties(ball["z"])


if __name__ == "__main__":
    instance = TwoCell([0, 0.5-10**(-8), 0, 0, -0.5, 0], 6, 2)
    instance.run()