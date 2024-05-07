from modelling.Simulator import Simulator
from modelling.ModelAB import ModelAB
from plot.Animator import Animator
from utils import get_data
from modelling.Fit.FitCurveFit import fit_model_whole
from modelling.Fit.FitMinimize import fit_fmin_model

#TODO
"""
SMT is messed up, check:
- equations on paper and in code
- angle calculations
- there should be a bug somewhere in the code that's causing these residuals to be wild
    - investigate by looking at the force components
    - dorsal is kinda weird 
    - there's 100% a shit that's wrong with the string equations


Fix Fit?
Fix Plots?


notes:
thing about the z plane and cells going beneath it
"""


# sim = Simulator(ModelAB, (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5), A=0, B=1.09679117, t_final=195)
# (4.683484906208419, 1.2174858680302194)
# 0.98309997, 1.09679117
# sim.run(True)
# SURFACES = [lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1]
data = get_data()
print(fit_fmin_model(get_data()))
# animator = Animator(sim.df)
# animator.animate()