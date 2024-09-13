import numpy as np
import matplotlib.pyplot as plt
import time

def execution_time(func):
    """
    Decorator to measure and print the time taken by a function to execute.

    Parameters:
        func (callable): The function to be decorated.

    Returns:
        callable: Decorated function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(f"Execution time of '{func.__name__}': {execution_time:.6f} seconds")
        return result
    return wrapper

def numerical_second_derivative(x, y):
    """
    Returns the array of the second derivative of an array of function values y and independent variable x

    Parameters:
    x (np.ndarray): An array of numbers representing values on the x-axis
    y (np.ndarray): An array of numbers of the function value dependent on x

    Returns:
    np.ndarray: An array of second derivative values
    """
    # Initialize the output array
    second_deriv = np.zeros_like(y)
    
    # Compute the second derivative using central difference method
    for i in range(1, len(x) - 1):
        second_deriv[i] = (y[i+1] - 2*y[i] + y[i-1]) / (0.5*(x[i+1] - x[i-1]))**2

    # Handle boundaries with forward and backward difference
    second_deriv[0] = (y[2] - 2*y[1] + y[0]) / (x[1] - x[0])**2
    second_deriv[-1] = (y[-1] - 2*y[-2] + y[-3]) / (x[-1] - x[-2])**2

    return second_deriv

def numerical_differentiator_array(x,y):

    # Returns the array of the numerical derivative of an array of function values y and independent variable x
    
    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : an array of numbers of the function value dependent on x

    if len(x)!=len(y):                                          # Checking array length compatibility
        raise Exception("Variable and function do not match")
    output=np.zeros_like(y)                                     # Output array initialization
    for i in range(len(x)):                                     # Differentiation loop
        if i-1==-1:
            m=(y[i+1]-y[i])/(x[i+1]-x[i])
        elif i+1==len(x):
            m=(y[i]-y[i-1])/(x[i]-x[i-1])
        else:
            m=(y[i+1]-y[i-1])/(x[i+1]-x[i-1])
        output[i]=m
    return output

def numerical_second_derivative2(x,y):
    
    # Returns the array of the second derivative of an array of function values y and independent variable x

    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : an array of numbers of the function value dependent on x

    first_deriv=numerical_differentiator_array(x,y)     # Calling first function derivative
    second_deriv=numerical_differentiator_array(x,first_deriv)  # Calling second function derivative
    return second_deriv

@execution_time
def numerical_integrator_array(x, y, c=0):
    """
    Returns the array of the numerical integral of an array of function values y and independent variable x.

    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        y (np.ndarray): Array of numbers representing function values dependent on x.
        c (float): Constant of integration (default is 0).

    Returns:
        np.ndarray: Array of numerical integral values.
    """
    if len(x) != len(y):  # Check array length compatibility
        raise ValueError("Lengths of x and y must match")
    
    output = np.zeros_like(y)  # Output array initialization
    s = 0  # Sum initialization
    
    for i in range(len(x)):  # Integration loop
        if i == 0:
            s += c
        else:
            s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])  # Trapezoidal rule
        output[i] = s
        
    return output

def unit_step(x, a):
    """
    Returns a Heaviside unit step function array from 0 to 1 at the boundary value a.

    Parameters:
        x (np.ndarray): Array of x values.
        a (float): Boundary value.

    Returns:
        np.ndarray: Heaviside unit step function array.
    """
    output = np.zeros_like(x)
    output[x >= a] = 1
    return output
@execution_time
def numerical_integrator_array2(x,y,c=0):
    # Returns the array of the numerical integral of an array of function values y and independent variable x
    
    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : an array of numbers of the function value dependent on x

    if len(x)!=len(y):                                          # Checking array length compatibility
        raise Exception("Variable and function do not match")   
    output=np.zeros_like(y)                                     # Output array initialization
    s=0                                                         # Sum initialization
    def mean(*nums):                                            # Average function
        return sum(nums)/len(nums)
    for i in range(len(x)):                                     # Integration loop
        if i==0:
            s+=c
        elif i+1==len(x):
            s+=(y[i])*(x[i]-x[i-1])
        else:
            s+=mean(y[i],y[i-1])*(x[i]-x[i-1])
        output[i]=s
    return output

@execution_time
def numerical_differentiator_array(x, y):
    """
    Returns the array of the numerical derivative of an array of function values y and independent variable x.

    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        y (np.ndarray): Array of numbers representing function values dependent on x.

    Returns:
        np.ndarray: Array of numerical derivative values.
    """
    if len(x) != len(y):  # Check array length compatibility
        raise ValueError("Lengths of x and y must match")
    
    output = np.zeros_like(y)  # Output array initialization
    
    # Using central differences for the interior points
    output[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    
    # Using forward difference for the first point
    output[0] = (y[1] - y[0]) / (x[1] - x[0])
    
    # Using backward difference for the last point
    output[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    return output

@execution_time
def numerical_differentiator_array1(x,y):

    # Returns the array of the numerical derivative of an array of function values y and independent variable x
    
    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   y       : an array of numbers of the function value dependent on x

    if len(x)!=len(y):                                          # Checking array length compatibility
        raise Exception("Variable and function do not match")
    output=np.zeros_like(y)                                     # Output array initialization
    for i in range(len(x)):                                     # Differentiation loop
        if i-1==-1:
            m=(y[i+1]-y[i])/(x[i+1]-x[i])
        elif i+1==len(x):
            m=(y[i]-y[i-1])/(x[i]-x[i-1])
        else:
            m=(y[i+1]-y[i-1])/(x[i+1]-x[i-1])
        output[i]=m
    return output

@execution_time
def numerical_differentiator(x, func):
    """
    Returns the array of the numerical derivative of a function f(x), dy/dx.
    
    Parameters:
        x (np.ndarray): Array of numbers representing values on the x axis.
        func (callable): A function taking x as an input.
    
    Returns:
        np.ndarray: Array of numerical derivative values.
    """
    y = func(x)  # Creating the function values
    output = np.zeros_like(y)  # Initializing an array
    
    # Calculate differences for the interior points using central difference
    dx = np.diff(x)
    output[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    
    # Forward difference for the first point
    output[0] = (y[1] - y[0]) / dx[0]
    
    # Backward difference for the last point
    output[-1] = (y[-1] - y[-2]) / dx[-1]
    
    return output

@execution_time
def numerical_differentiator1(x,func):

    # Returns the array of the numerical derivative of a function f(x), dy/dx
    
    # Inputs:           x       : an array of numbers representing values on the x axis
    #                   func    : a function taking x as an input

    y=func(x)                                       # Creating the function values
    output=np.zeros_like(y)                         # Initializing an array                            
    for i in range(len(x)):                         # Differentiation loop
        if i-1==-1:
            m=(y[i+1]-y[i])/(x[i+1]-x[i])
        elif i+1==len(x):
            m=(y[i]-y[i-1])/(x[i]-x[i-1])
        else:
            m=(y[i+1]-y[i-1])/(x[i+1]-x[i-1])
        output[i]=m
    return output

@execution_time
def square_wave(x, frequency=1, amplitude=1, phase=0, offset=0):
    """
    Generates a square wave based on the input parameters.
    
    Parameters:
    x (np.ndarray): Array of time or independent variable values.
    frequency (float): Frequency of the square wave. Default is 1.
    amplitude (float): Amplitude of the square wave. Default is 1.
    phase (float): Phase shift of the wave. Default is 0.
    offset (float): Vertical offset of the wave. Default is 0.
    
    Returns:
    np.ndarray: Array of square wave values.
    """
    # Calculate the sine wave
    sine_wave = np.sin(2 * np.pi * frequency * x + phase)
    
    # Generate the square wave
    square_wave = amplitude * np.sign(sine_wave) + offset
    
    return square_wave

def sine_wave(x, frequency=1, amplitude=1, phase=0, offset=0):
    """
    Generates a sine wave based on the input parameters.
    
    Parameters:
    x (np.ndarray): Array of time or independent variable values.
    frequency (float): Frequency of the sine wave. Default is 1.
    amplitude (float): Amplitude of the sine wave. Default is 1.
    phase (float): Phase shift of the wave. Default is 0.
    offset (float): Vertical offset of the wave. Default is 0.
    
    Returns:
    np.ndarray: Array of sine wave values.
    """
    # Precompute constant to avoid repetition in the calculation
    angular_frequency = 2 * np.pi * frequency
    
    # Calculate the sine wave
    y = amplitude * np.sin(angular_frequency * x + phase) + offset
    
    return y

def triangle_wave(x, frequency=1, amplitude=1, phase=0, offset=0):
    
    """
    Generates a triangle wave based on the input parameters.
    
    Parameters:
    x (np.ndarray): Array of time or independent variable values.
    frequency (float): Frequency of the triangle wave. Default is 1.
    amplitude (float): Amplitude of the triangle wave. Default is 1.
    phase (float): Phase shift of the wave. Default is 0.
    offset (float): Vertical offset of the wave. Default is 0.
    
    Returns:
    np.ndarray: Array of triangle wave values.
    """
    # Compute the period of the wave
    period = 1 / frequency
    
    # Compute the wave's fractional part and adjust for phase shift
    fractional_part = np.mod((x - phase), period) / period
    
    # Compute the absolute distance from the midpoint and scale
    y = 4 * amplitude * np.abs(fractional_part - 0.5) - amplitude
    
    # Apply the vertical offset
    y += offset
    
    return y

def sawtooth_wave(x, frequency=1, amplitude=1, phase=0, offset=0):
    """
    Generates a sawtooth wave based on the input parameters.
    
    Parameters:
    x (np.ndarray): Array of time or independent variable values.
    frequency (float): Frequency of the sawtooth wave. Default is 1.
    amplitude (float): Amplitude of the sawtooth wave. Default is 1.
    phase (float): Phase shift of the wave. Default is 0.
    offset (float): Vertical offset of the wave. Default is 0.
    
    Returns:
    np.ndarray: Array of sawtooth wave values.
    """
    # Calculate the normalized phase
    normalized_phase = (x - phase) * frequency
    
    # Compute the sawtooth wave using the modulo operation
    y = amplitude * (normalized_phase - np.floor(normalized_phase))
    
    # Apply the vertical offset
    y += offset
    
    return y

@execution_time
def RC_circuit(t, v, R, C):
    """
    Simulate the transient response of an RC circuit to an input signal.
    
    Parameters:
        t (np.ndarray): Array of time points.
        v (np.ndarray): Input signal (e.g., voltage).
        R (float): Resistance value of the resistor (ohms).
        C (float): Capacitance value of the capacitor (farads).
    
    Returns:
        np.ndarray: Simulated output signal of the RC circuit.
    """
    # Calculate time step
    dt = np.diff(t)
    dt = np.mean(dt) if len(dt) > 0 else 1.0
    
    # Calculate time constant
    tau = R * C
    
    # Initialize output array
    out = np.zeros_like(v)
    
    # Initialize previous time and previous output
    t_prev = t[0]
    y_res = 0
    
    # Iterate over time points
    for i in range(1, len(t)):
        t_curr = t[i]
        
        # Calculate response of RC circuit using trapezoidal rule
        response = (v[i] - y_res) * (t_curr - t_prev) / tau
        
        # Update previous time
        t_prev = t_curr
        
        # Update response
        y_res += response
        
        # Store response in output array
        out[i] = y_res
    
    return out

# Define time array
t = np.arange(0, 10, 0.01)

# Define input signal (e.g., voltage)
voltage = np.sin(t)

# Resistance and capacitance values
R = 1000  # 1 kohm
C = 2.5e-2  # 1 uF


# Example usage:
x = np.arange(0, 100, 0.01)
# FFy= numerical_second_derivative(x, y)
def func(x):
    y=sine_wave(x,0.25)
    z=np.zeros_like(y)
    z[y>=0]= y[y >= 0]
    return z

y=func(x)
v=RC_circuit(x,y,R,C)


@execution_time
def plot_2d():
    print('plotting')
    # plt.axis([0, 1, z_min, z_max])
    plt.grid()
    plt.plot(x, y, color='red', label='y(x)')
    plt.plot(x, v, color='blue',label='y\'(x)')
    # plt.plot(x, second_derivative_values, color='magenta',label='y\'\'(x)')
    plt.legend()
    plt.show()

plot_2d()
