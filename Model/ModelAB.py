from Model.Model import Model

class ModelAB(Model):
    def __init__(self, A, B, system) -> None:
        super().__init__(A, B)

    def step(self, time):
        super().step(time)
        