from __future__ import print_function
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
import numpy as np



set_log_level(WARNING)

p11=0.0
p12=0.0
p13=0.0
p21=1.0
p22=1.0
p23=2.0
nx=10
ny=10
nz=10

# No-slip boundary
class NoslipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and not (x[2] < DOLFIN_EPS  or x[2] > p23 - DOLFIN_EPS)

# Inflow boundary
class InflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        #return x[2] < DOLFIN_EPS and on_boundary
        return x[2] < DOLFIN_EPS and on_boundary and not (x[0] > 1 - DOLFIN_EPS or x[0] < DOLFIN_EPS or x[1] > 1 - DOLFIN_EPS or x[1] < DOLFIN_EPS )

# Outflow boundary
class OutflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        #return x[2] > p23 - DOLFIN_EPS and on_boundary
        return x[2] > p23 - DOLFIN_EPS and on_boundary and not (x[0] > 1 - DOLFIN_EPS or x[0] < DOLFIN_EPS or x[1] > 1 - DOLFIN_EPS or x[1] < DOLFIN_EPS )

#---------------------------------------------------------------------

#Create Geometry


mesh = BoxMesh(Point(p11,p12,p13), Point(p21,p22,p23), nx, ny, nz)   #UnitCubeMesh(10, 10, 10)

#mesh = BoxMesh(Point(0.0,0.0,0.0), Point(1.0,1.0,1.0),  12, 12, 60)  
h = mesh.hmin()

print("h=", h)

# Plot Mesh 
plot(mesh, interactive=True)

 #! problem specific
Time = 5.0
f = Constant((0., 0., 0.))     # force
Se = Constant(0.01)              # energy source
nu = Constant(1./8.)           # kinematic viscosity  
lamda = Constant(0.58)          # Thermal Conductivity
Cp = Constant(4200.)           # Specific Heat Capacity
dt = 0.01/1.                  # time step CFL with 1 = max. velocity  
k = Constant(dt)               # time step                     # total simulation time
u_0 = Expression(('0.', '0.', '0.'))    # initial velocity
p_0 = Expression('0.')              # initial pressure
T_0 = Expression('303.')            # initial temperature = 303 K

  #! solver specific
V = VectorFunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)


u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
T = TrialFunction(Q)
Tq = TestFunction(Q)  

u0= Function(V)
u_00 = project(u_0, V)
#u0 = assign(u0, u_00) 
 
p0= Function(Q)
p_00 = project(p_0, Q)
#p0 = assign(p0, p_00) 

T0= Function(Q)
T_00 = project(T_0, Q)
#T0 = assign(T0, T_00) 

us = Function(V)
u1 = Function(V)
p1 = Function(Q)  
T1 = Function(Q)


# tentative velocity
F0 = (1./k)*inner(u - u0, v)*dx + inner(dot(grad(u0), u0), v)*dx\
     + nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a0, L0 = system(F0)


  # projection
F1 = inner(grad(p), grad(q))*dx + (1./k)*q*div(us)*dx
a1, L1 = system(F1)

  # finalize
F2 = (1./k)*inner(u - us, v)*dx + inner(grad(p1), v)*dx 
a2, L2 = system(F2)

  # Energy
F3 = (1./k)*inner(T - T0, Tq)*dx + inner(dot(grad(T0), u0), Tq)*dx\
       + (lamda/Cp)*inner(grad(T), grad(Tq))*dx - (1./Cp)*inner(Se, Tq)*dx 
a3, L3 = system(F3)

  # boundary conditions
  # Walls
b_v = DirichletBC(V, Constant((0.0, 0.0, 0.0)), NoslipBoundary())
#b_T = DirichletBC(Q, Constant(0.), OutflowBoundary())       # Walls T = 323 K
                                                            #Neuman Condotions for temperature at wall

  # Inlet
b_v1 = DirichletBC(V, Constant((0.0, 0.0, 1.0)), InflowBoundary())  # Inlet velocity
b_T1 = DirichletBC(Q, Constant(323.) , InflowBoundary())      # Inlet T = 303 K

  # Outlet
b_p1 = DirichletBC(Q, p_00, OutflowBoundary()) #Pressure at outlet boundary
b_v2 = DirichletBC(V, Constant((0.0, 0.0, 1.0)), OutflowBoundary()) # Outlet Velocity (same as inlet)


#Collect Boundary Conditions

bcs_v = [b_v, b_v1, b_v2]
bcs_p = [b_p1]
bcs_T = [b_T1]


A0 = assemble(a0)
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

solver02 = KrylovSolver('gmres', 'ilu')
solver1 = KrylovSolver('cg', 'petsc_amg')
solver3 = KrylovSolver('gmres', 'ilu')

ufile = File("3Dresults/Box_chorin_u.pvd")
pfile = File("3Dresults/Box_chorin_p.pvd")
Tfile = File("3Dresults/Box_chorin_T.pvd")

iter = 0
t = 0
while t < Time:
      t += dt
      iter += 1

      b = assemble(L0)
      [bc.apply(A0, b) for bc in bcs_v]
      solver02.solve(A0, us.vector(), b)

      b = assemble(L1)
      [bc.apply(A1, b) for bc in bcs_p]
      solver1.solve(A1, p1.vector(), b)

      b = assemble(L2)
      [bc.apply(A2, b) for bc in bcs_v]
      solver02.solve(A2, u1.vector(), b)

      b = assemble(L3)
      [bc.apply(A3, b) for bc in bcs_T]
      solver1.solve(A3, T1.vector(), b)

      # Plot solution
      plot(p1, title="Pressure", rescale=True)
      plot(u1, title="Velocity", rescale=True)
      plot(T1, title="Temperature", rescale=True)

      ufile << u1
      pfile << p1
      Tfile << T1

      u0.assign(u1)
      p0.assign(p1)
      T0.assign(T1)
      print(t)
      print(iter)
  
      t+= dt

# Hold plot
plot(mesh, interactive=True)

print(u0, p1, T0)
