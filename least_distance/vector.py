from sympy import symbols, lambdify, diff, Lambda



def nearest_point(coord: tuple[float, float, float], surface: Lambda):
    pass



x, y, z = symbols('x y z ')
p = x, y, z
sphere1 = x**2 + y**2 + z ** 2 - 1


der = lambdify((x,y,z), diff(sphere1, x))

print(der(1,2,3))




