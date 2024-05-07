from modelling.Simulator import Simulator
from modelling.ModelAB import ModelAB
from modelling.ModelExtendingSpring import ModelExtendingSpring
from plot.Animator import Animator
from utils import get_data, get_cortical_data
from modelling.Fit.FitCurveFit import fit_model_whole
from modelling.Fit.FitMinimize import fit_fmin_model
from modelling.Fit.FitCoritcalFlow import fit_cortical

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

# SURFACES = [lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1]
data = get_data()
print(fit_fmin_model(get_data()))
# 0.70911644, 0.79862217

# sim = Simulator(ModelAB, (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5), A=A, B=B, t_final=195)
# sim = Simulator(ModelExtendingSpring, (0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, -0.5, 0, -0.5, 0.5, 0), 
#                 A=0.01, B=1, t_final=195, surfaces=[lambda x: (2*x[0]/3)**2 + x[1]**2 + x[2]**2 - 1])
# sim.run(True)
# animator = Animator(sim.df)
# animator.animate()



# data = get_cortical_data()
# fit_cortical(data)

