"""Start up flow parameters"""
from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


dt = 0.025 #Time Stepping [s]
Tf = 1.0
e = Constant('5')
n = Constant('50')
#f = Expression(('1.0-exp(-zet*t)'), t=0.0, zet=zet)
#g = Expression(('0.5*(1+(exp(2*(zet*t-net))-1)/(exp(2*(zet*t-net))+1))'), t=0.0,  zet=zet, net=net)

x=list()
y=list()
z=list()
t=0.0
while t < Tf + DOLFIN_EPS:
      print(0.5*(1.0+tanh(e*t-2.5)))
      print(1.0-exp(-e*t))
      x.append(t)
      y.append(0.5*(1.0+tanh(e*t-2.5)))
      z.append(1.0-exp(-e*t))
      t+=dt

plt.plot(x, y, 'r-', label='f')
plt.show()
plt.plot(x, z, 'b-', label='g')
plt.show()



