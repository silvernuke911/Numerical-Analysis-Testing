import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NUMERICAL CALCULUS FUNCTIONS
def numerical_differentiator(x,y):
    # Returns the numerical derivative of a function y(x)
    if len(x)!=len(y):
        raise Exception("Variable and function do not match")
    output=np.array([])
    for i in range(len(x)):
        
        if i-1==-1:
            m=(y[i+1]-y[i])/(x[i+1]-x[i])
        elif i+1==len(x):
            m=(y[i]-y[i-1])/(x[i]-x[i-1])
        else:
            m=(y[i+1]-y[i-1])/(x[i+1]-x[i-1])
        output=np.append(output,m)
    return output
def numerical_integrator(x,y,c=0):
    # Returns the numerical integral of a function y(x)
    if len(x)!=len(y):
        raise Exception("Variable and function do not match")
    output=np.array([])
    s=0
    def mean(*nums):
        return sum(nums)/len(nums)
    for i in range(len(x)):
        if i==0:
            s+=c
        elif i==len(x):
            s+=(y[i])*(x[i]-x[i-1])
        else:
            s+=mean(y[i],y[i-1])*(x[i]-x[i-1])
        output=np.append(output,s)
    return output

def numerical_second_derivative(x,y):
    first_deriv=numerical_differentiator(x,y)
    second_deriv=numerical_differentiator(first_deriv)
    return second_deriv

def numerical_second_integral(x,y,c1=0,c2=0):
    first_integral=numerical_integrator(x,y,c1)
    second_integral=numerical_integrator(first_integral,c2)
    return second_integral

# LISTING FUNCTIONS
def arrays_to_list(*arrays):
    # Convert the arbitrary arguments to a list
    arrays_list = list(arrays)
    return arrays_list
def save_to_csv(header,arrays):
    # Saves the arrays to a csv file
    filename='test_results.csv'
    data={}
    for i in range(len(header)):
        data[names[i]]=arrays[i]
    data=pd.DataFrame(data,columns=header)
    print(data)
    data.to_csv(filename,index=False)

# MATH FUNCTIONS
def unit_step(x,a):
    # creates a heaviside unit step function from 0 to 1 at the boundary a
    y=0
    output=np.array([])
    for i in x:
        if i<a:
            y=0
        if i>=a:
            y=1
        output=np.append(output,y)
    return output
def test_func(x):
    # custom function for testing
    out=np.array([])
    y=0
    for i in x:
        if i>0 and i<2:
            y=3*i
        elif i>=2 and i<7:
            y=6
        elif i>=7:
            y=-2*i+20
        out=np.append(out,y)
    return out
def connecting_function(arr, x1, x2):
    """
    Create a function that smoothly transitions from y=0 to y=1 
    on the interval [x1, x2].
    
    Parameters:
        arr (np.ndarray): Array of values where the function will be evaluated.
        x1 (float): Lower boundary of the transition interval.
        x2 (float): Upper boundary of the transition interval.
    
    Returns:
        np.ndarray: Array of values representing the smoothly transitioning function.
    """
    # Inner function psi(x): Monotone increasing function from [0, 1]
    def psi(x):
        return x ** 3
    # Inner function phi(x): Smoothly interpolates between 0 and 1 in the interval [x1, x2]
    def phi(x):
        alph = (x - x1) / (x2 - x1)
        return psi(alph) / (psi(alph) + psi(1 - alph))
    # Initialize the output array with zeros
    out = np.zeros_like(arr)
    # Indices where x is between x1 and x2
    mask = (arr >= x1) & (arr < x2)
    # Calculate values using phi function for indices in the mask
    out[mask] = phi(arr[mask])
    # Set values to 1 where x >= x2
    out[arr >= x2] = 1
    
    return out
def lin_floor_func(x,a=1):
    # returns a floor function divided into a parts per unit
    return np.floor(x*a)/a
def norm_dist(x,stddev=1,mean=0):
    # returns a normal distribution with standard deviation and mean as parameters
    coeff=1/(stddev*np.sqrt(2*np.pi))
    expo=-0.5*((x-mean)/stddev)**2
    return coeff*np.exp(expo)
def double_slit_experiment(x,slit_distance=0.05,slit_width=0.05,wavelength=0.0000650,screen_distance=100):
    # returns an intensity function of the double slit experiment
    theta=x/screen_distance
    def sinc_term(x):
        return np.sinc(np.pi*slit_width*np.sin(x)/wavelength)
    def cos_term(x):
        return np.cos(np.pi*slit_distance*np.sin(x)/wavelength)
    return (cos_term(theta)**2)*(sinc_term(theta)**2)

# ELECTRONICS FUNCTIONS
def square_wave(x,frequency=1,amplitude=1,phase=0,offset=0):
    # square wave
    y=amplitude*np.sign(np.sin(2*np.pi*frequency*x+phase))+offset
    return y
def sine_wave(x,frequency=1,amplitude=1,phase=0,offset=0):
    # sine function
    y=amplitude*np.sin(2*np.pi*frequency*x+phase)+offset
    return y
def triangle_wave(x,frequency=1,amplitude=1,phase=0,offset=0):
    # triangular function
    y=(4*amplitude*frequency*np.absolute(np.mod(((x-phase)-1/(4*frequency)),1/frequency)-1/(2*frequency))-amplitude)+offset
    return y   
def sawtooth_wave(x,frequency=1,amplitude=1,phase=0,offset=0):
    def saw(x):
        return x-np.floor(x)
    return amplitude*saw((x-phase)*frequency)+offset
def RC_circuit(x,y,R,C):
    tau=R*C
    y_res=0
    out=np.zeros_like(y)
    def rc_func(y,y_res,tau,t,t_prev):
        e=(y-y_res)*(t-t_prev)
        return e/tau
    for i in range(len(x)):
        v=y[i]
        t=x[i]
        t_prev=x[i-1]
        response=rc_func(v,y_res,tau,t,t_prev)
        y_res+=response
        out[i]=y_res
    return out
def PID_controller(x,y_desired,y_initial,K_p,K_i,K_d):

    lag=0.1
    y_response=y_initial
    error=0
    error_prev=0
    integral=0
    t=x[0]
    t_prev=x[0]
    response=0
    control_list=np.array([])
    error_list=np.array([])

    def PID(y_desired,y_measured,K_p,K_i,K_d):
        nonlocal error,error_prev,integral

        if (t-t_prev)==0:
            return [0,0]
        
        # print(y_desired,y_measured)
        error=y_desired-y_measured
        P=K_p*error

        integral=integral+K_i*error*(t-t_prev)
        I=integral
        # print(error,error_prev)
        # print(error-error_prev,t-t_prev)
        D=K_d*(error-error_prev)/(t-t_prev)
        error_prev=error
        return [P+I+D, error ]

    for i in range(len(x)):
        n=PID(y_desired[i],y_response,K_p,K_i,K_d)
        print
        response=n[0]
        y_response+=response
        t=x[i]
        t_prev=x[i-1]
        control_list=np.append(control_list,y_response)
        error_list=np.append(error_list,n[1])
        print(t,t_prev,y_desired[i],response,y_response)
    return control_list, error_list

def func(x):
    return square_wave(x,0.25)

x_min = 0
x_max = 10
x = np.arange(x_min,x_max,0.01)
y = func(x)

def ylims(y):
    yran=(np.max(y)-np.min(y))
    yave=(np.max(y)+np.min(y))/2
    return [yave-0.75*yran,yave+0.75*yran]
y_min = ylims(y)[0]
y_max = ylims(y)[1]

d_y  = numerical_differentiator(x,y)
F_y  = numerical_integrator(x,y)
dd_y = numerical_differentiator(x,d_y)
FF_y = numerical_integrator(x,F_y)

arrays=arrays_to_list(x,y,d_y,F_y)
names=['x','y','d_y','F_y']
save_to_csv(names,arrays)

def plot():
    print('Currently plotting function')
    plt.axis([x_min,x_max, y_min, y_max])
    plt.plot(x,y,color='red')
    # plt.plot(x,y2,color='blue')
    # plt.plot(x,y3,color='green')
    plt.plot(x,d_y,color='blue')
    plt.plot(x,F_y,color='green')
    # plt.plot(x,dd_y)
    # plt.plot(x,FF_y)
    plt.grid()
    plt.show()
    return
plot()


# PID Boilerplate
# pid=PID_controller(x,y,0,0.5,0.8,0.01)
# y2=pid[0]
# y3=pid[1]

# RC Circuit Boilerplate
# R,C=1250,0.0005
# y2=RC_circuit(x,y,R,C)
# y3=1-np.exp(-((x-1)/(R*C)))

# Smooth connectiong function boiler plate
# f1=func1(x)
# f2=func2(x)
# return f1*(1-connecting_function(x,-2,2))+f2*(connecting_function(x,-2,2))