"""Nonlinear PDE problem example code. Newton Method"""


from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
import numpy as np


#Define Domain

mesh = UnitSquareMesh(35, 35)

#Define Subdomains

class LowerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and  (x[0] < DOLFIN_EPS)

class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and  (x[0] > 1 - DOLFIN_EPS)

# Define Problem Parameters

m = 2
def q(u):
    return (1+u)**m

#Define Function Spaces
V= FunctionSpace(mesh, "CG", 1)

#Define Boundary Conditions

w=Expression("x[0]*x[0]+x[1]*x[1]",degree=1)

bc1 = DirichletBC(V, Constant(0.0), LowerBoundary())
bc2 = DirichletBC(V, w, UpperBoundary())

bcs = [bc1, bc2]

# Define variational problem for initial guess (q(u)=1, i.e., m=0)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(nabla_grad(u), nabla_grad(v))*dx
f = Constant(0.0)
L = f*v*dx
A, b = assemble_system(a, L, bcs)
u_k = Function(V)
U_k = u_k.vector()
solve(A, U_k, b)

#Define derivative of q
def Dq(u):
    return m*(1+u)**(m-1)

#Define Variational problem in du
du = TrialFunction(V) # u = u_k + omega*du
a = inner(q(u_k)*nabla_grad(du), nabla_grad(v))*dx + \
    inner(Dq(u_k)*du*nabla_grad(u_k), nabla_grad(v))*dx
L = -inner(q(u_k)*nabla_grad(u_k), nabla_grad(v))*dx

du = Function(V)
u  = Function(V)  # u = u_k + omega*du

#Boundary Conditions for Variational problem in du

Gamma_0_du = DirichletBC(V, Constant(0.0), LowerBoundary())
Gamma_1_du = DirichletBC(V, Constant(0.0), UpperBoundary())
bcs_du = [Gamma_0_du, Gamma_1_du]

plot(mesh)

omega = 1.0       # relaxation parameter
eps = 1.0
tol = 1.0E-10
iter = 0
maxiter = 25
while eps > tol and iter < maxiter:
      iter += 1
      A, b = assemble_system(a, L, bcs_du)
      solve(A, du.vector(), b)
      eps = np.linalg.norm(du.vector().array(), ord=np.Inf)
      print'Norm:', eps
      u.vector()[:] = u_k.vector() + omega*du.vector()
      plot(u, title='u', rescale=False, interactive=True)
      u_k.assign(u)


