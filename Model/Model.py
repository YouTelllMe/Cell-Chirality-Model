from Cell import FourCellSystem

class Model:
    def __init__(self, A, B, system: FourCellSystem) -> None:
        self.A = A
        self.B = B
        self.system = system

    def step(self, time) -> None:
        return
