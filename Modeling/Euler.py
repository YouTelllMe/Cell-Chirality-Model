import numpy as np
import pandas as pd
from Model.Model import Model
import config


class Euler:

    def __init__(self, model: Model) -> None:
        self.model = model
        # takes 400 steps (initial position vector inclusive)
        N = config.MODEL_STEPS
        # t=0 not included in tau
        self.tau = np.linspace(1/N, 1, N-1)


    def run(self, save: bool) -> None:
        """
        Perform euler's method by repeatedly calling the given function. Note that this function
        is programmed to handle systems of 12 equations (4 position vectors in succession)
        """
        # initial position (t=0)
        model_snapshots = [self.model]

        for time in self.tau:
            model_snapshots.append(model_snapshots[-1].step(time))

        position_coord = [model.system.get_position() for model in model_snapshots]
        distance = {
            "12": [model.system.info.dist_12 for model in model_snapshots], 
            "13": [model.system.info.dist_13 for model in model_snapshots], 
            "14": [model.system.info.dist_14 for model in model_snapshots],
            "23": [model.system.info.dist_23 for model in model_snapshots], 
            "24": [model.system.info.dist_24 for model in model_snapshots], 
            "34": [model.system.info.dist_34 for model in model_snapshots]
            }
        angle = {"dorsal1": [model.system.info.ang_dorsal1 for model in model_snapshots], 
                "dorsal2": [model.system.info.ang_dorsal2 for model in model_snapshots], 
                "anterior1": [model.system.info.ang_anterior1 for model in model_snapshots],
                "anterior2": [model.system.info.ang_anterior2 for model in model_snapshots]}

        self.euler_df = pd.DataFrame(position_coord)
        self.distance_df = pd.DataFrame(distance)
        self.angle_df = pd.DataFrame(angle)

        if save:
            self.save_Dataframe()
            self.animate()

        return (self.euler_df, self.distance_df, self.angle_df)

    def save_Dataframe(self):
        """
        Saves the distance, angle, and position data from euler's method. 
        """
        try:
            self.euler_df.to_csv(config.POSITION_DATAPATH, index=False)
            self.distance_df.to_csv(config.DISTANCE_DATAPATH, index=False)
            self.angle_df.to_csv(config.ANGLES_DATAPATH, index=False)
        except:
            print("Dataframe doesn't exist")
        

    def animate(self):
        EulerAnimator(self).animate()