""" The STEADY state incompressible Navier-Stokes solved using an Iterative Newton Method"""
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
import numpy as np


#Define Domain

mesh = UnitSquareMesh(15, 15)

#Define Subdomains

class NoslipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary 

class LowerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and  (x[0] < DOLFIN_EPS)

class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and  (x[0] > 1 - DOLFIN_EPS)

# Define Problem Parameters

Re = 10.0
f = Constant(('0.0', '0.0'))

#Define Function Spaces
Q = FunctionSpace(mesh, "CG", 2)
V= VectorFunctionSpace(mesh, "CG", 1)

W = MixedFunctionSpace([V, Q])

#Define Functions
p0= Function(Q)
p = TrialFunction(Q)
q = TestFunction(Q)
p_k = Function(Q)
u0= Function(V)
u = TrialFunction(V)
v = TestFunction(V)
u_k = Function(V)

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

w0 = Function(W)
w = Function(W)
w_k = Function(W)



# Bilinear Forms
def a(x, y): 
    return inner(nabla_grad(x), nabla_grad(y))*dx #nonlinear term
def b(x, y):
    return inner((dot(u_k, nabla_grad(x)) + dot(x, nabla_grad(u_k))), y)*dx   
def c(x, y):
    return inner(grad(x), grad(y))*dx
def d(x, y):
    return inner(div(x),y)*dx
def ds(x, y):
    return inner(grad(x), y)*dx
def e(x, y):
    return inner(inner(nabla_grad(x), nabla_grad(x)), y)*dx

#Linear Functionals
M1 = inner(-(Re*dot(u_k, nabla_grad(u_k))+ grad(p_k) + grad(div(u_k)) + f),v)*dx
M2 = 0.0   #=inner(div(f), q)*dx


#Define Boundary Conditions """CHANGE TO NEWTON!!!!!!!!!!!!!!!!!!!!!!!!!"""

bcu = DirichletBC(W.sub(0), Constant(('0.0', '0.0')), NoslipBoundary())
bcp = DirichletBC(W.sub(1), Constant(1.0), UpperBoundary())

bcs = [bcu, bcp]

# Define variational problem for initial guess
a0 = a(u, v) + ds(p, v)
L0 = inner(f,v)*dx

a1=d(u,q)
L1=0.0

a2=c(p, q)
L2=M2 

a_= a0+a1+a2
L= L0+L1+L2
 
solve(a_ == L, w0, bcs)
print(w.vector())

(u0, p0) = w0.split(True)     # Extract u_k and p_k from w_k

u_k.assign(u0)
p_k.assign(p0)


#Define Variational problem in du
du = TrialFunction(V) # u = u_k + omega*du
dp = TrialFunction(Q)
dw = Function(W)

a0 = a(du, v) + Re*b(du, v) + ds(dp, v) 
L0 = M1

a1= d(du, q)
L1 = 0.0

a2 = c(dp, q) - Re*e(du, q)
L2 = M2

a_ = a0+a1+a2
L = L0+L1+L2


#Boundary Conditions for Variational problem in du

Gamma_0_dp = DirichletBC(Q, Constant(0.0), NoslipBoundary())
Gamma_0_du = DirichletBC(V, Constant(('0.0', '0.0')), NoslipBoundary())

bcs_du = [Gamma_0_du, Gamma_0_dp]

omega = 1.0       # relaxation parameter
eps = 1.0
tol = 1.0E-10
iter = 0
maxiter = 25
while eps > tol and iter < maxiter:
      iter += 1
      solve(a_ == L, dw, bcs_du)   # A, b = assemble_system(a_, L, bcs_du)
      (du, dp) = dw.split(True)                           
      epsu = np.linalg.norm(du.vector().array(), ord=np.Inf)
      epsp = np.linalg.norm(dp.vector().array(), ord=np.Inf)
      print'Norm:', eps
      u.vector()[:] = u_k.vector() + omega*du.vector()
      p.vector()[:] = p_k.vector() + omega*dp.vector()
      plot(u, title='Velocity', rescale=True)
      plot(p, title='Pressure', rescale=True)
      u_k.assign(u)
      p_k.assign(p)

