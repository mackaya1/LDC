from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt, fabs
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
import matplotlib.mlab as mlab



# Define Geometry
B=1
L=1
x0 = 0.0
y0 = 0.0
x1 = B
y1 = L
m=5
nx=m*B
ny=m*L

c = min(x1-x0,y1-y0)

mesh= RectangleMesh(Point(x0,y0), Point(x1, y1), nx, ny)

V = VectorFunctionSpace(mesh, "CG", 1)

u = Expression(('x[0]*x[0]','x[1]*x[1]'), degree=1)

v = Expression((('1.0','1.0'),('1.0','1.0')), degree=1)

u = interpolate(u,V)
#v = interpolate(v,V)

plot(u,interactive=True)

a= assemble(inner(grad(u),v)*dx)
b= assemble(inner(nabla_grad(u),v)*dx)

print a.vector().array()


