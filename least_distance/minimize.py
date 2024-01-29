from scipy.optimize import minimize


a = 10
b = 50
c = 100

fun = lambda x: ((x[0] - x[3])**2 + (x[1] - x[4])**2 + (x[2] - x[5])**2)**(1/2)
surface1 = lambda x: x[0]**2 + x[1]**2 + x[2]**2 -1
# surface2 = lambda x: x[5]-10
# surface2 = lambda x: x[3]**2/a**2 + x[4]**2/b**2 + x[5]**2/c**2 - 1
surface2 = lambda x: x[3]**2/a**2 + x[4]**2/b**2 + x[5]**2/c**2 - 1

x0 = (0,0,0,0,0,0)

cons = ({'type': 'eq', 'fun': surface1},
        {'type': 'eq', 'fun': surface2})

res = minimize(fun, x0, constraints = cons, method = 'Newton-CG')

print(res)




