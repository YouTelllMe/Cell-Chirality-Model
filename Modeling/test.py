from Simulator import Simulator
from ModelAB import ModelAB






sim = Simulator(ModelAB, (0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, -0.5, 0, -0.5, 0.5, 0), A=1, B=1)
sim.run(True)