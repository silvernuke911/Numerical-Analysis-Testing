import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['figure.dpi']= 300
# plt.rcParams['savefig.dpi'] = 300
# find the function where dy/dx = k y

# I suppose the main sulution for numerical integrals is that the next form of the function is y+dy, and dy = y*dx
# dx = 0.0001
# x_min,x_max = 0,2

# x = np.arange(x_min,x_max,dx)
# y = np.zeros_like(x)
# z=np.exp(x)
# y_i=1
# for i in range(len(x)):
#     if i==0:
#         y[0]=y_i
#     else:
#         y_i=y_i+y_i*dx
#         y[i]=y_i

# find the function y'=-y^2

# dx = 0.0001
# x_min,x_max = 0,2

# x = np.arange(x_min,x_max,dx)
# y = np.zeros_like(x)
# z =(x+1)**(-1)
# y_i = 1
# for i in range(len(x)):
#     if i==0:
#         y[0]=y_i
#     else:
#         y_i=y_i-y_i**2 * dx
#         y[i]=y_i

# find the function y''=-y^2 
# dx = 0.01
# x_min,x_max = 0,2

# x = np.arange(x_min,x_max,dx)
# y = np.zeros_like(x)
# y_i = 1
# z = np.exp(x)
# dydx = lambda x,y: y

# for some differential equation y'=f(x,y)
def runge_kutta(x, dx, y0, func):
    n = len(x)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        xi = x[i-1]
        yi = y[i-1]
        k1 = func(xi, yi)
        k2 = func(xi + dx / 2, yi + dx * k1 / 2)
        k3 = func(xi + dx / 2, yi + dx * k2 / 2)
        k4 = func(xi + dx, yi + dx * k3)
        y[i] = yi + (dx / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y
def Forward_Euler(x,dx,y_i,func):
    y = np.zeros_like(x)
    y[0]=y_i
    for i in range(1,len(x)):
        y[i] = y[i - 1] + dx * func(x[i - 1], y[i - 1])
    return y
# def double_diff(x, dx, y_i, y_ii, func):
#     """
#     Solve the second-order differential equation y'' = f(y', y, x) using the Runge-Krutta method.

#     Parameters:
#     x (array-like): Array of x values.
#     dx (float): Average spacing between the x values.
#     y_i (float): Initial y value (y at x[0]).
#     y_ii (float): Initial y' value (dy/dx at x[0]).
#     func (function): Function that takes (y_prime, y, x) and returns y''.

#     Returns:
#     np.ndarray: Array containing the numerical solution for y at each x.
#     """
#     n = len(x)
#     y = np.zeros(n)  # Array to store y values
#     y_prime = np.zeros(n)  # Array to store y' values

#     y[0] = y_i
#     y_prime[0] = y_ii

#     # Function to update y_prime using Runge-Kutta method
#     def func_prime(x, y_prime):
#         return func(y_prime, y, x)

#     for i in range(1, n):
#         # Update y' using Runge-Krutta
#         y_prime_values = Runge_Krutta(x[:i+1], dx, y_prime[i-1], func_prime)
#         y_prime[i] = y_prime_values[-1]

#         # Update y using Runge-Krutta
#         y_values = Runge_Krutta(x[:i+1], dx, y[i-1], lambda x, y: y_prime[i-1])
#         y[i] = y_values[-1]

#     return y

def double_diff(x, dx, y_i, y_ii, func):
    n = len(x)
    y = np.zeros(n)
    y_prime = np.zeros(n)

    y[0] = y_i
    y_prime[0] = y_ii

    for i in range(1, n):
        # Calculate k values for y'
        k1_y_prime = y_prime[i-1]
        k2_y_prime = y_prime[i-1] + 0.5 * dx * func(y_prime[i-1], y[i-1], x[i-1])
        k3_y_prime = y_prime[i-1] + 0.5 * dx * func(y_prime[i-1] + 0.5 * dx * k2_y_prime, y[i-1] + 0.5 * dx * k1_y_prime, x[i-1] + 0.5 * dx)
        k4_y_prime = y_prime[i-1] + dx * func(y_prime[i-1] + 0.5 * dx * k3_y_prime, y[i-1] + 0.5 * dx * k3_y_prime, x[i-1] + dx)

        y[i] = y[i-1] + (dx / 6) * (k1_y_prime + 2 * k2_y_prime + 2 * k3_y_prime + k4_y_prime)

        # Calculate k values for y''
        k1_y = func(y_prime[i-1], y[i-1], x[i-1])
        k2_y = func(y_prime[i-1] + 0.5 * dx * k1_y, y[i-1] + 0.5 * dx * k1_y_prime, x[i-1] + 0.5 * dx)
        k3_y = func(y_prime[i-1] + 0.5 * dx * k2_y, y[i-1] + 0.5 * dx * k2_y_prime, x[i-1] + 0.5 * dx)
        k4_y = func(y_prime[i-1] + dx * k3_y, y[i-1] + dx * k3_y_prime, x[i-1] + dx)

        y_prime[i] = y_prime[i-1] + (dx / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
    
    return y

# find the function y''=-y^2 
dx = 0.001
x_min,x_max = 0,2

x = np.arange(x_min,x_max,dx)
y = np.sin(x)
y_i = 0
y_ii = 1

z2 = double_diff(x,dx,y_i,y_ii,lambda y_prime,x,y: -y)


def plot():
    plt.plot(x,y)
    plt.plot(x,z2)
    plt.grid()
    plt.show()
plot()
    