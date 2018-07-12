"""Countour plot for the Poisson Equation solved over the unit square"""
"""The Poisson Equation is solved using FEniCS LinearVariationalSolver """


from dolfin import *
from math import pi, sin, cos, sqrt
import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Create mesh and define function space

nx = 30
ny = 30

mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, "Lagrange", 1)

x = interpolate(Expression("x[0]"),V)
y = interpolate(Expression("x[1]"),V)

n = (nx+1)*(ny+1)
for i in range(n):
    print(i , (x.vector().array()[i],y.vector().array()[i]))

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Expression("sin(5*x[0])")
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)
# Plot solution
plot(u, interactive=True)

print((u.vector().array()))

x_coords = np.linspace(0.05, 0.95, 100)
y_coords = np.zeros(len(x_coords))
z_coords = np.zeros(len(x_coords))

for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    z_coords[i] = u(x, y)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='2d')
ax.plot(x_coords, y_coords, z_coords)
plt.show() 

