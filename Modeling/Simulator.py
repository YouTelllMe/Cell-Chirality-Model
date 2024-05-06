import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


class Simulator:

    def __init__(self, model, y0, **kwargs) -> None:
        self.fun = lambda t, y: model.get_velocity(t, y, A = kwargs['A'], B = kwargs['B'])
        self.y0 = y0
        self.TAU_INITIAL = 0
        self.TAU_FINAL = 1 # non-dimensionalized

    def run(self, save: bool) -> None:
        """
        Uses RK45

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        """
        solver = solve_ivp(self.fun, 
                           [self.TAU_INITIAL, self.TAU_FINAL], 
                           self.y0, 
                           t_eval = np.linspace(self.TAU_INITIAL, self.TAU_FINAL, 40)
                           )

        while solver.status == "running":
            solver.step() # last step included

        format_y = np.transpose(np.array(solver.y))
        df = pd.DataFrame(format_y, columns=[str(index) for index in range(12)], index=solver.t)


        if save:
            df.to_excel('output.xlsx')
        # print(df)


        # position_coord = [model.system.get_position() for model in model_snapshots]
        # distance = {
        #     "12": [model.system.info.dist_12 for model in model_snapshots], 
        #     "13": [model.system.info.dist_13 for model in model_snapshots], 
        #     "14": [model.system.info.dist_14 for model in model_snapshots],
        #     "23": [model.system.info.dist_23 for model in model_snapshots], 
        #     "24": [model.system.info.dist_24 for model in model_snapshots], 
        #     "34": [model.system.info.dist_34 for model in model_snapshots]
        #     }
        # angle = {"dorsal1": [model.system.info.ang_dorsal1 for model in model_snapshots], 
        #         "dorsal2": [model.system.info.ang_dorsal2 for model in model_snapshots], 
        #         "anterior1": [model.system.info.ang_anterior1 for model in model_snapshots],
        #         "anterior2": [model.system.info.ang_anterior2 for model in model_snapshots]}

        # self.euler_df = pd.DataFrame(position_coord)
        # self.distance_df = pd.DataFrame(distance)
        # self.angle_df = pd.DataFrame(angle)

        # if save:
        #     self.save_Dataframe()
        #     self.animate()

        # return (self.euler_df, self.distance_df, self.angle_df)

    # def save_Dataframe(self):
    #     """
    #     Saves the distance, angle, and position data from euler's method. 
    #     """
    #     try:
    #         self.euler_df.to_csv(config.POSITION_DATAPATH, index=False)
    #         self.distance_df.to_csv(config.DISTANCE_DATAPATH, index=False)
    #         self.angle_df.to_csv(config.ANGLES_DATAPATH, index=False)
    #     except:
    #         print("Dataframe doesn't exist")
        

    # def animate(self):
    #     EulerAnimator(self).animate()