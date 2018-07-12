from dolfin import *

ny = 15
 
# Set up problem by loading mesh from file
mesh = Mesh('/home/alexmackay/Desktop/dolfin-2.xml.gz')

#mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), nx, ny)

plt = plot(mesh, interactive=True, scale=2.0)

# Use a constant forcing field to drive the flow from right to left
f = Expression(('-1.',' 0.'))

# Set Problem Parameters
h = mesh.hmin()
dt = 0.01*h/1.
k= Constant(dt)
nu = Constant(1./8.) 
Se = Constant(0.0)
Cp = Constant(4000)
kappa = Constant(10.)
u_0 = Expression(('0.', '0.'))  # initial velocity
p_0 = Expression('1.')     # initial pressure on Right Wall
p_1 = Expression('0.') #Initial Pressure on Left Wall
c_0 = Expression('330.')   # Temperature on the Dolfin 
c_1 = Expression('0.')  # Temperature 

# Function Spaces
Q = FunctionSpace(mesh, "Lagrange", 1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# No-slip boundary condition for velocity on the dolfin 1
dolfin = AutoSubDomain(lambda x, on_boundary: on_boundary and not
                       (near(x[0], 0) or near(x[0], 1.) or near(x[1], 0.) or near(x[1], 1.)))

# No-slip boundary condition for velocity on the dolfin 2
class NoslipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and not (x[0] < DOLFIN_EPS or x[1] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS)

# Wall Boundary
class WallBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] < DOLFIN_EPS or x[1] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS)

# Right Wall Boundary (INLET)
class InflowBoundary(SubDomain):
      def inside(self, x, on_boundary):
          return (x[0] < x1 - DOLFIN_EPS)

# Left Wall Boundary (OUTLET)
class OutflowBoundary(SubDomain):
      def inside(self, x, on_boundary):
          return (x[0] < DOLFIN_EPS)


# Set up variational form.
v_p = TestFunction(Q)
v_u = TestFunction(V)
Tq = TestFunction(Q)


u = TrialFunction(V)
p = TrialFunction(Q)
c = TrialFunction(Q)

us = Function(V)

u0 = Function(V)
u1 = Function(V)
p0= Function(Q)
p1 = Function(Q)
c0 = Function(Q)
c1 = Function(Q)

U = 0.5*(u0 + u1)
C = 0.5*(c0 + c1) 

u_00 = project(u_0, V)
#u0 = assign(u0, u_00) 
p_00 = project(p_0, Q)
#p0 = assign(p0, p_00) 
c_00 = project(c_0, Q)
#T0 = assign(T0, T_00) 


#Tentative velocity 
F0 = (1.0/dt)*inner(u - u0, v_u)*dx + \
     inner(dot(u0, nabla_grad(u0)), v_u) + \
     nu*inner(grad(u), grad(v_u))*dx - \
     inner(p, div(v_u))*dx - \
     inner(f, v_u)*dx
a0, L0 = system(F0)

# projection
F1 = inner(div(U), v_p)*dx - inner(grad(p), grad(v_p))*dx + (1./k)*v_p*div(us)*dx
a1, L1 = system(F1)

  # finalize
F2 = (1./k)*inner(u - us, v_u)*dx + inner(grad(p1), v_u)*dx 
a2, L2 = system(F2)


#Termal balance
F3 = (1.0/dt)*inner(c - c0, v_c)*dx + \
     inner(dot(U_, grad(C)), v_c)*dx + \
     nu*(1. + c_1**2)*inner(grad(C), grad(v_c))*dx
     
a3, L3 = system(a3)

L0 = system(a0)

#Boundary Conditions

bu = DirichletBC(V, u_0, dolfin) # Using the second dolfin boundary definition 
bp0 = DirichletBC(Q, p_0, InflowBoundary()) #
bp1 = DirichletBC(Q, p_1, OutflowBoundary())
bT0 = DirichletBC(Q, T_0, dolfin)
bT1 = DirichletBC(Q, T_1, WallBoundary()) # Wall Temperature

bcs_u = [bu]
bcs_p = [bp0] #Presuure Boundary Condition
bcs_T = [bT0] #Only Dolfin Temperature boundary condition used  

A0 = assemble(a0)
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

solver02 = KrylovSolver('gmres', 'ilu')
solver1 = KrylovSolver('cg', 'petsc_amg')
solver3 = KrylovSolver('gmres', 'ilu')

ufile = File("Thermresults/thermal_u.pvd")
pfile = File("Thermresults/thermal_chorin_p.pvd")
Tfile = File("Thermresults/thermal_T.pvd")

iter = 0
t = 0
while t < Time:
      t += dt
      iter += 1

      b0 = assemble(L0)
      [bc.apply(A0, b0) for bc in bcs_u]
      solver02.solve(A0, us.vector(), b0)

      b1 = assemble(L1)
      [bc.apply(A1, b1) for bc in bcs_p]
      solver1.solve(A1, p1.vector(), b1)

      b2 = assemble(L2)
      [bc.apply(A2, b2) for bc in bcs_u]
      solver02.solve(A2, u1.vector(), b2)

      b3 = assemble(L3)
      [bc.apply(A3, b3) for bc in bcs_T]
      solver3.solve(A3, c1.vector(), b3)

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



