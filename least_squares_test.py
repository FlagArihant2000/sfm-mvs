from scipy.optimize import least_squares
import numpy as np

def fun_rosenbrock(x):
	return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])
	
x0_rosenbrock = np.array([2,2])

res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)

print(res_1)

print(fun_rosenbrock(x0_rosenbrock))
