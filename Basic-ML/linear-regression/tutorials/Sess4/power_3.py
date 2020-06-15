# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-7, 8, .4)

def f(x):
    return 2.0 * pow(x, 3.0) - 12.0 * x + 12.0


def dfx(x):
    return 6.0 * pow(x,2.0) - 12.0

def dfx_roots():
    roots = []
    for i in np.arange(-7,8,1):
       #print(dfx(i))
       if abs(dfx(i)) < .000001:
           roots.append(i)
    return roots

# plt.figure(1)
plt.plot(x, f(x), 'g')
plt.plot(x, dfx(x), 'c--')
dx = np.arange(-7, 8, 2)

for i in np.arange(-7,10,3):
    x_i = np.arange(i - 1, i + 1, .5)
    m = dfx(i)    
    c =  f(i) - m*i
    y_i = m*(x_i)  +  c
    plt.plot(x_i,y_i,'b-')



plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Change in gradient example')
plt.legend(('f(x)',"f'(x)",'gradient'))
plt.grid(axis='both',color='c', alpha=0.25)
plt.show()
