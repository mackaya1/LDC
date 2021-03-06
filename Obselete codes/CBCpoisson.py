
from __future__ import print_function
from mshr import *
from math import pi, sin, cos, sqrt
from ufl import *
from dolfin import *
from cbc.pdesys import *
from cbc.pdesys import *

mesh = UnitSquareMesh(100, 100) 
Q = FunctionSpace(mesh, 'CG', 1) 
u = TrialFunction(Q) 
v = TestFunction(Q) 
u_ = Function(Q) 
f = Constant(1.) 
F = inner(grad(u), grad(v))*dx + f*v*dx
bcs = DirichletBC(Q, (0.), 'on_boundary')

# Implementation with LinearVariationalProblem/Solver
a, L = lhs(F), rhs(F)
poisson_problem = LinearVariationalProblem(a, L, u_, bcs=bcs)
poisson_solver  = LinearVariationalSolver(poisson_problem)
poisson_solver.solve()
plot(u_, title="Velocity", rescale=True, interactive=True)


#Implementation with cbc.pdesys
poisson = PDESubSystem(vars(), ['u'], bcs=[bcs], F=F)
poisson.solve()


"""mesh = UnitSquareMesh(10, 10)
# Change desired items in the problem_parameters dict from cbc.pdesys
problem = Problem(mesh, problem_parameters)
poisson = PDESystem([['u']], problem, solver_parameters) # Creates FunctionSpace, Functions etc.
poisson.f = Constant(1.)

class Poisson(PDESubSystem):
    def form(self, u, v_u, f, **kwargs):    # v_u is the TestFunction
        return inner(grad(u), grad(v_u))*dx + f*v_u*dx

bcs = DirichletBC(poisson.V['u'], (0.), DomainBoundary())
poisson.pdesubsystems['u'] = Poisson(vars(poisson), ['u'], bcs=[bcs])
problem.solve()"""
