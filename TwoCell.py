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

    def __init__(self, cur_pos, spring_constant) -> None:
        self.k = spring_constant
        self.cur_pos = np.array(cur_pos)
        self.cell1 = Cell([cur_pos[0], cur_pos[1], cur_pos[2]])
        self.cell2 = Cell([cur_pos[3], cur_pos[4], cur_pos[5]])
        self.euler = [cur_pos]


        self.STEPS = 500
        self.H = 1/self.STEPS
        # self.FORCE_T = lambda t: 10*t*math.e**-(5*t)
        self.FORCE_T = lambda t: 0.5
        self.SURFACE = lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 1.5


    def get_velocity(self, t):
        cell1 = np.array([self.cur_pos[0], self.cur_pos[1], self.cur_pos[2]])
        cell2 = np.array([self.cur_pos[3], self.cur_pos[4], self.cur_pos[5]])
        
        u12 = (cell1-cell2) / np.linalg.norm(cell1-cell2) # unit vector from 2 to 1 (pushes 1 away from 2)
        u21 = (cell2-cell1) / np.linalg.norm(cell2-cell1) # unit vector from 1 to 2 (pushes 2 away from 1)

        cell1_prime = u12 * self.FORCE_T(t)
        cell2_prime = u21 * self.FORCE_T(t)

        cell1_prime += self.cell_wall_step(cell1)
        cell2_prime += self.cell_wall_step(cell2)

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

    def cell_wall_step(self, pos):
        min_point = find_min(pos, self.SURFACE)
        norm = np.linalg.norm(min_point.x-pos)
        if norm < 0.5:  
            return np.array(self.k * (0.5 - norm) * (min_point.x-pos)/norm)
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
        cell_dfs[0].to_csv(os.path.join(os.getcwd(), "cell2.csv"))
        
        positions_vectors = []
        animated_indicies = range(0, 499, 10) # 50 steps animated
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
    

    def generate_ball(self, position: tuple[float, float, float], radius: float):
        """
        Return evenly spaced coordinates on the sphere with center at position with radius radius.
        """
    
        delta = np.pi / 20
        rotation_matrix = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])

        center_x, center_y, center_z = position
        curr_x, curr_y, curr_z = 0, 0, radius
        ref_z = 0
        x = [center_x+curr_x]
        y = [center_y+curr_y]
        z = [center_z+curr_z]

        for _ in range(20):
            phi_vector = np.matmul(rotation_matrix, [ref_z, curr_z])
            ref_z = phi_vector[0]
            curr_z = phi_vector[1]
            curr_x = ref_z
            curr_y = 0
            for _ in range(40):
                theta_vector = np.matmul(rotation_matrix, [curr_x, curr_y])
                curr_x = theta_vector[0]
                curr_y = theta_vector[1]
                x.append(center_x+curr_x)
                y.append(center_y+curr_y)
                z.append(center_z+curr_z)
        
        return {"x": x, "y": y, "z": z}


if __name__ == "__main__":
    instance = TwoCell([0, 0, 0.5, 0, 0, -0.5], 1)
    instance.run()