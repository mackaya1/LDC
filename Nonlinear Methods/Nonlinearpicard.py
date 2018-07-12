"""Nonlinear PDE problem example code. PICARD ITERATION"""


from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
import numpy as np


#Define Domain

mesh = UnitSquareMesh(25, 25)

#Define Subdomains

class LowerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and  (x[0] < DOLFIN_EPS)

class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and  (x[0] > 1 - DOLFIN_EPS)

# Define Problem Parameters

m = 7
def q(u):
    return (1+u)**m

#Define Function Spaces
V= FunctionSpace(mesh, "CG", 1)

# Define variational problem for Picard iteration
u = TrialFunction(V)
v = TestFunction(V)
u_k = Function(V)
u_k = project(Constant(0.0), V)  # previous (known) u
a = inner(q(u_k)*nabla_grad(u), nabla_grad(v))*dx
f = Constant(0.0)
L = f*v*dx


#Define Boundary Conditions

bc1 = DirichletBC(V, Constant(0.0), LowerBoundary())
bc2 = DirichletBC(V, Constant(1.0), UpperBoundary())

bcs = [bc1, bc2]

# Picard iterations
u = Function(V)     # new unknown function
eps = 1.0           # error measure ||u-u_k||
tol = 1.0E-10        # tolerance
iter = 0            # iteration counter
maxiter = 25        # max no of iterations allowed
while eps > tol and iter < maxiter:
    iter += 1
    solve(a == L, u, bcs)
    plot(u, title='u', rescale=True)
    diff = u.vector().array() - u_k.vector().array()
    eps = np.linalg.norm(diff, ord=np.Inf)
    print 'iter=%d: norm=%g' % (iter, eps)
    u_k.assign(u)   # update for next iteration

