from Simulator import Simulator
from ModelAB import ModelAB






sim = Simulator(ModelAB, (0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, -0.5, 0, -0.5, 0.5, 0), A=1, B=1, t_final=195)
sim.run(False)


SURFACES = [lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1]