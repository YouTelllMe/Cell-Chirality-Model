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



class FourCell:

    def __init__(self, cur_pos, spring_constant, a, c) -> None:
        self.a = a # controls growing springs (it's growth rate as a factor of integrate of cortical flow)
        self.c = c # strength of friction / coritcal flow
        self.k = spring_constant
        self.cur_pos = np.array(cur_pos)
        self.euler = [cur_pos]

        self.STEPS = 2000
        self.H = 1/8 # 1/500; why do we do h = 1/ steps again? 
        # self.FORCE_T = lambda t: 10*t*np.e**-(5*t)
        # self.FORCE_T = lambda t: 0.5
        # self.SURFACES = [lambda x: x[1] - 1, lambda x: x[1] + 1]
        self.SURFACES = [lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1]

        
    def get_velocity(self, t):
        print(self.cur_pos)
        ABal = np.array([self.cur_pos[0], self.cur_pos[1], self.cur_pos[2]])
        ABar = np.array([self.cur_pos[3], self.cur_pos[4], self.cur_pos[5]])
        ABpr = np.array([self.cur_pos[6], self.cur_pos[7], self.cur_pos[8]])
        ABpl = np.array([self.cur_pos[9], self.cur_pos[10], self.cur_pos[11]])
        
        distance12 = np.linalg.norm(ABal-ABar)
        distance13 = np.linalg.norm(ABal-ABpr)
        distance14 = np.linalg.norm(ABal-ABpl)
        distance23 = np.linalg.norm(ABar-ABpr)
        distance24 = np.linalg.norm(ABar-ABpl)
        distance34 = np.linalg.norm(ABpr-ABpl)

        u12 = (ABal-ABar) / distance12 # unit vector from 2 to 1 
        u13 = (ABal-ABpr) / distance13 # unit vector from 3 to 1
        u14 = (ABal-ABpl) / distance14 # unit vector from 4 to 1
        u23 = (ABar-ABpr) / distance23 # unit vector from 3 to 2
        u24 = (ABar-ABpl) / distance24 # unit vector from 4 to 2
        u34 = (ABpr-ABpl) / distance34 # unit vector from 4 to 3

        
        cortical_flow_r = 0.000345*t*np.e**(-0.012732*t)
        cortical_flow_l = 0.00071*t*np.e**(-0.0166*t)
        cortical_integral = ((1-(0.014666*t+1)*np.e**(-0.014666*t))/0.014666**2)

        # 0.014666 is the avg lambda of left right cortical flow 
        ABal_prime = self.k * (-u12 * (distance12 - (1 + self.a * cortical_integral)) 
                              - u14 * (distance14 - 1)) + self.c * cortical_flow_l * (np.cross(-u14, -u12) - np.cross(u14, u34))

        ABar_prime = self.k * (u12 * (distance12 - (1 + self.a * cortical_integral)) 
                              - u23 * (distance23 - 1)) + self.c * cortical_flow_r * (np.cross(-u23, u12) - np.cross(u23, -u34))

        ABpr_prime = self.k * (- u34 * (distance34 - (1 + self.a * cortical_integral)) 
                                      + u23 * (distance23 - 1)) + self.c * cortical_flow_r * (np.cross(u23, -u34) - np.cross(-u23, u12))

        ABpl_prime = self.k * (u34 * (distance34 - (1 + self.a * cortical_integral)) 
                              + u14 * (distance14 - 1)) + self.c * cortical_flow_l * (np.cross(u14, u34) - np.cross(-u14, -u12))

        # cell wall forces 
        for surface in self.SURFACES:
            ABal_prime += self.cell_wall_step(ABal, surface)
            ABar_prime += self.cell_wall_step(ABar, surface)
            ABpr_prime += self.cell_wall_step(ABpr, surface)
            ABpl_prime += self.cell_wall_step(ABpl, surface)
            
        return np.concatenate((ABal_prime, ABar_prime, ABpr_prime, ABpl_prime))
    
    def step(self, t):
        self.cur_pos = self.cur_pos + self.get_velocity(t) * self.H
        self.euler.append(self.cur_pos)

    def run(self):
        time = np.linspace(1, 250, self.STEPS-1)
        for t in time:
            self.step(t)

        self.df = pd.DataFrame(self.euler)
        self.animate()

    def cell_wall_step(self, pos, surface):
        min_point = find_min(pos, surface)
        norm = np.linalg.norm(min_point.x-pos)
        if norm < 0.5:  
            return np.array(self.k * (0.5 - norm) * (pos-min_point.x)/norm)

            # Van der Waals
            # return np.array(self.k * ((0.5/(0.5-norm))**12 - (0.5/(0.5-norm))**6) * (pos-min_point.x)/norm)
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
        cell_dfs[0].to_csv(os.path.join(os.getcwd(), "ABal.csv"))
        cell_dfs[1].to_csv(os.path.join(os.getcwd(), "ABar.csv"))
        cell_dfs[2].to_csv(os.path.join(os.getcwd(), "ABpr.csv"))
        cell_dfs[3].to_csv(os.path.join(os.getcwd(), "ABpl.csv"))
        
        positions_vectors = []
        animated_indicies = range(0, self.STEPS-1, 20)
        labels = ["ABal","ABar","ABpr","ABpl"]
        for position_index in range(len(cell_dfs)):
            new_position, = AX.plot([],[],[],".", alpha=0.4, markersize=3, label=labels[position_index])
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
        
        # plt.show()

        # save animation from different angles
        anim.save(os.path.join(os.getcwd(), "4-cell", "4cell-1.png"))
        AX.view_init(90, -90, 0)
        anim.save(os.path.join(os.getcwd(), "4-cell", "4cell-2.png"))
        AX.view_init(0, -90, 0)
        anim.save(os.path.join(os.getcwd(), "4-cell", "4cell-3.png"))
        AX.view_init(0, 0, 0)
        anim.save(os.path.join(os.getcwd(), "4-cell", "4cell-4.png"))
        

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
    instance = FourCell([0.5, 0.5, 0, 0.5, -0.5, 0, 
                         -0.5, -0.5, 0, -0.5, 0.5, 0], 1, 0.00001, 1)
    instance.run()