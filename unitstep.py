import random as rnd
import math as mt
import matplotlib.pyplot as plt


def x_generator(start, end, step):
    decimals=len(str(step))-1
    steps=round((end-start)/step)
    x=start-step
    x_list=[]
    for i in range(0,steps+1):
        x+=step
        x_list.append(round(x,decimals))
    return(x_list)

def function_list_maker(x_list,f_x):
    output=[]
    for i in range(len(x_list)):
        output.append(f_x(x_list[i]))
    return output

def unit_step_function(x,a):
    #0 <= x and 
    if x<a:
        return 0
    if x>=a:
        return 1
    
def plotter():
    plt.axis([x_start,x_end,y_start,y_end])
    plt.grid()
    plt.plot(
        x_list,
        function_list1,
        color="red",
        marker="",
        linestyle="-"
    )
    plt.plot(
        x_list,
        function_list2,
        color="orange",
        marker="",
        linestyle="-"
    )
    plt.plot(
        x_list,
        function_list3,
        color="magenta",
        marker="",
        linestyle="-"
    )
    plt.plot(
        x_list,
        unit_func_list,
        color="blue",
        marker="",
        linestyle="-"
    )
    plt.show()


def func(x):
    return x**2

def unit_func(x):
    return g_1(x)*unit_step_function(x,a)+g_2(x)*unit_step_function(x,b) 


def g_1(x):
    return 2*mt.sin(x)
def g_2(x):
    return mt.cos(x)
def g_3(x):
    return g_1(x)+g_2(x)

def f(x):
    if x<a:
        return g_1(x)
    if x>=a:
        return g_2(x)
    
x_start=0
x_end=6
a=2
b=1
x_list=x_generator(x_start,x_end,0.01)
function_list1=function_list_maker(x_list,g_1)
function_list2=function_list_maker(x_list,g_2)
function_list3=function_list_maker(x_list,g_3)
unit_func_list=function_list_maker(x_list,unit_func)

y_start=min(function_list1)-1
y_end=max(function_list1)+1

for i in range(len(x_list)):
    print(x_list[i],function_list1[i],function_list2[i],unit_func_list[i])
plotter()
