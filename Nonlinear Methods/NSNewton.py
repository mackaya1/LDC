""" The incompressible Navier-Stokes solved using an Iterative Newton Method with a FIRST order discretisation in space used"""

from dolfin import *
from mshr import *
from math import *
import numpy as np
import pylab as pl


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

plot(mesh, interactive=True, axes=True)

#Define Function Spaces
Q = FunctionSpace(mesh, "Lagrange", 1)
V=  VectorFunctionSpace(mesh, "Lagrange", 1)
W = MixedFunctionSpace([V, Q])

# Define Problem Parameters
Re = 10.0
dt= 0.01
T = 2.0
k= dt                                                          #f = Expression(('0.0', '0.0'))
f1 = Expression(('0.1', '0.0'))#, element = W.sub(0).ufl_element())
df= div(f1)

#Define Functions
u0=Function(V)
##u=Function(V)
p0=Function(Q)
#p=Function(Q)


#w = Function(W)
w_k = Function(W)

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

#Define Boundary Conditions 

bcu = DirichletBC(W.sub(0), Constant(('0.0', '0.0')), NoslipBoundary())
bcu1 = DirichletBC(W.sub(0), Constant(('1.0', '0.0')), LowerBoundary())
bcp = DirichletBC(W.sub(1), Constant('1000.0'), NoslipBoundary())

bcs = [bcu, bcu1, bcp]

# Loop for time dependent problem


t = dt
while t < T:
      
      u, p = TrialFunctions(W)
      w = Function(W)      

      # Define variational problem for initial guess (w0 ----> w_k)
      a0 = (inner(grad(u),grad(v)) - inner(p,div(v)))*dx     #a(u, v) + ds(p, v) """inner(nabla_grad(u), nabla_grad(v))"""
      L0 = dot(f1,v)*dx

      a1=inner(div(u), q)*dx  #Divergence Free Condition
      L1=0.0

      a2=inner(grad(p), grad(q))*dx  #Poisson Equation Needed
      L2=0.0

      a_= a0+a1+a2
      L= L0+L1+L2

      A, b = assemble_system(a_, L, bcs) #Solve Linear Variational Problem
      solve(A, w_k.vector(), b)

      for i in w_k.vector().array():  # Eliminate numerical noise
          if  i < 1.0e-10:
              i = 0.0

      u_k, p_k = w_k.split(True) # Extract u_k and p_k from w_k

       #We need to redefine u_k as a vector to use in iterative part of the Newton method

      # Bilinear Forms
      def n(x, y): 
          return inner(x, y)*dx #nonlinear term
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
      M1 = inner(-(Re*dot(u_k, nabla_grad(u_k))+ grad(p_k) + grad(div(u_k)) + f1),v)*dx + (Re/k)*inner(u0, v)*dx
      M2 = inner(div(u_k), q)*dx  #=inner(div(f), q)*dx


      #Define Variational problem in du
      du, dp = TrialFunctions(W) # u = u_k + omega*du , p = p_k + omega*dp
      dw = Function(W)

      a0 = (Re/dt)*n(u,v) + a(du, v) + Re*b(du, v) + ds(dp, v) 
      L0 = M1

      a1= d(du, q)
      L1 = -M2

      a2= c(dp, q) #- Re*inner(div(grad(du)*du), q)*dx #Re*e(du, q)
      L2= -inner((f1+grad(p_k)), grad(q))*dx + inner(div(grad(u_k)*u_k), q)*dx

      a_ = a0+a1+a2
      L = L0+L1+L2

      #Boundary Conditions for Variational problem in du

      Gamma_0_dp = DirichletBC(Q, Constant(0.0), NoslipBoundary())
      Gamma_0_du = DirichletBC(V, Constant(('0.0', '0.0')), NoslipBoundary())
  
      bcs_du = [Gamma_0_du, Gamma_0_dp]

      omega = 1.0       # relaxation parameter
      epsu= 1.0
      epsp= 1.0
      tol = 1.0E-10
      iter = 0
      maxiter = 25
      while (epsu + epsp) > tol and iter < maxiter:
            iter += 1
            A, b = assemble_system(a_, L, bcs_du)
            solve(A, dw.vector(), b)
            du, dp = dw.split(True)  
            #print('du=', du.vector().array())
            #print('dp=', dp.vector().array())                          
            epsu = np.linalg.norm(du.vector().array(), ord=np.Inf)
            epsp = np.linalg.norm(dp.vector().array(), ord=np.Inf)
            print'Norm:', epsu 
            print'Norm:', epsp 
            u, p= w.split(True)
            u.vector()[:] = u_k.vector() + omega*du.vector()
            p.vector()[:] = p_k.vector() + omega*dp.vector()
            plot(u, title='Velocity', rescale=True)
            plot(p, title='Pressure', rescale=True)
            #print('u_k=', u_k.vector().array())
            #print('p_k=', p_k.vector().array())
            w_k.assign(w)
            u_k.assign(u)
            p_k.assign(p)

      #Move to next time step
      t += dt
      print("t =", t)
      u0.assign(u)
      p0.assign(p)

