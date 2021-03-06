"""INCOMPRESSIBLE TAYLOR GALERKIN METHOD"""


from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt, fabs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.mlab as mlab

# MATPLOTLIB CONTOUR FUNCTIONS
def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells()) # Mesh Diagram

def mplot(obj):                     # Function Plot
    plt.gca().set_aspect('equal')
    if isinstance(obj, Function):
        mesh = obj.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise(AttributeError)
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            plt.tripcolor(mesh2triang(mesh), C)
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')


# Define Geometry
B=1
L=1
x0 = 0.0
y0 = 0.0
x1 = B
y1 = L
m=60
nx=m*B
ny=m*L

mesh= RectangleMesh(Point(x0,y0), Point(x1, y1), nx, ny)

print('Number of Cells:', mesh.num_cells())
print('Number of Vertices:', mesh.num_vertices())


# Mesh Refine Code

for i in range(3):
      g = (max(x1,y1)-max(x0,y0))*0.05/(i+1)
      cell_domains = CellFunction("bool", mesh)
      cell_domains.set_all(False)
      for cell in cells(mesh):
          x = cell.midpoint()
          if  fabs(x[0]) < g and x[1] > L-g or x[0] > B-g and x[1] > L-g: 
              cell_domains[cell]=True
      #plot(cell_domains, interactive=True)
      mesh = refine(mesh, cell_domains, redistribute=True)
      print('Number of Cells:', mesh.num_cells())
      print('Number of Vertices:', mesh.num_vertices())

plot(mesh, interactive=True)

#Define Boundaries 
                                                                          
class Omega0(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] > y1 - 10*DOLFIN_EPS and on_boundary  else False   


class Omega1(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] < x0 + x1*DOLFIN_EPS  or x[0] > x1 - x1*DOLFIN_EPS or x[1] < y0 + DOLFIN_EPS and on_boundary else False 

omega1= Omega1()
omega0= Omega0()

class Omega2(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] > 1.0 - DOLFIN_EPS and x[0] < DOLFIN_EPS and on_boundary or x[1] < DOLFIN_EPS and x[0] < DOLFIN_EPS and on_boundary else False 

class Omega3(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] > 1.0 - DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary or x[1] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary else False   

omega2= Omega2()
omega3= Omega3()


# MARK SUBDOMAINS (Create mesh functions over the cell facets)
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(5)
omega0.mark(sub_domains, 0)
omega1.mark(sub_domains, 3)
#omega2.mark(sub_domains, 2)
#omega3.mark(sub_domains, 3)
plot(sub_domains, interactive=True)



# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 2)
W = TensorFunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
#rho=TrialFunction(Q)
p = TrialFunction(Q)
T = TrialFunction(Q)
S = TrialFunction(W)
v = TestFunction(V)
q = TestFunction(Q)
R = TestFunction(W)


# Set parameter values
h = mesh.hmin()
dt = 0.01  #Time Stepping   
Tf = 15.0    #Final Time
cp = 4000.0
g=Expression(('0','-100'))
mu_0 = 5.0*10E-2
T_0 = 300.0#Reference temperature
T_h = 350.0
C=120.0 #Sutherland's constant
Rc = 3.33*10E1
kappa = 25.0
heatt= 5.0
Pth=100000.0
rho=1000.0
c=1500.0 #Speed of Sound

print(rho)


#Define Discretised Functions

u0=Function(V)
us=Function(V)
u12=Function(V)
u1=Function(V)
rho0=Function(Q)
rho1=Function(Q)
p00=Function(Q)
p0=Function(Q)
p1=Function(Q)
mu=Function(Q)
T0=Function(Q)
T1=Function(Q)
tau0=Function(W)
tau14=Function(W)
tau12=Function(W)
tau1=Function(W)

# Define FLuid Parameters
Re=15.0
mu=mu_0#*(T0+C)/(T_0+C)*(T0/T_0)**(3/2)

kappa = 25.0
heatt= 5.0

alpha = 1.0/(rho*cp)
k = Constant(dt)



boundary_parts = FacetFunction("size_t", mesh)
omega0.mark(boundary_parts,0)
omega1.mark(boundary_parts,1)
ds = Measure("ds")[boundary_parts]


# Define boundary conditions
td= Constant('5')
e = Constant('6')
w=Expression('T_0+0.5*(1.0+tanh(e*t-2.5))*(T_h-T_0)', t=0.0, e=e, T_0=T_0, T_h=T_h)
sl=Expression(('0.5*(1+tanh(e*t-2.5))*5.0','0.0'), t=0.0, e=e)
pp=Expression('0.5*(1.0+tanh(e*t-2.5))*000.0', t=0.0, e=e)
uin=Expression(('-4*(x[1]-y0)*(x[1]-y1)', '0.0'), y0=y0, y1=y1)
uout=Expression(('-(x[1]-y0)*(x[1]-y1)+5', '0.0'), y0=y0, y1=y1)

 # Dirichlet Boundary Conditions 
inflow = DirichletBC(V, uin, omega0)
outflow = DirichletBC(V, (1.0,0.0), omega1)
noslip  = DirichletBC(V, (0.0, 0.0), omega1)  # No Slip boundary conditions on the left part of the flow wall
slip  = DirichletBC(V, sl, omega0)  # Slip boundary conditions on the second part of the flow wall 
temp0 =  DirichletBC(Q, 100.0, omega0)    #Dirichlet Boundary Condition on the inner bearing 
press1 = DirichletBC(Q, pp, omega0)
press2 = DirichletBC(Q, 0.0, omega1)

#Collect Boundary Conditions
bcu = [noslip, slip]
bcp = []
bcT = [temp0]
bcS = []

"""# Initial Density Field
rho_array = rho0.vector().array()
for i in range(len(rho_array)):  
    rho_array[i] = rho_0
rho0.vector()[:] = rho_array """

# Initial Temperature Field
T_array = T0.vector().array()
for i in range(len(T_array)):  
    T_array[i] = T_0
T0.vector()[:] = T_array  



#Define Strain Rate and other tensors
sr = 0.5*(grad(u0) + transpose(grad(u0)))
gamdot = inner(sr,grad(u0))

# TAYLOR GALERKIN METHOD

# Weak Formulation 

#Half Step
a1=(Re/(dt/2.0))*inner(u,v)*dx+0.25*inner(grad(u),grad(v))*dx
L1=(Re/(dt/2.0))*inner(u0,v)*dx-0.25*inner(grad(u0),grad(v))*dx-Re*inner(grad(u0)*u0,v)*dx+2.0*inner(p0,div(v))*dx-inner(p00,div(v))*dx#+inner(g,v)*dx

#Predicted U* Equation
a2=(Re/dt)*inner(u,v)*dx+0.25*inner(grad(u),grad(v))*dx
L2=(Re/dt)*inner(u0,v)*dx-0.25*inner(grad(u12),grad(v))*dx-Re*inner(grad(u12)*u12,v)*dx+2.0*inner(p0,div(v))*dx-inner(p00,div(v))*dx

#Continuity Equation 1
a3=inner(grad(p),grad(q))*dx
L3=inner(grad(p0),grad(q))*dx+0.5*(Re/dt)*inner(us,grad(q))*dx

#Velocity Update
a5=(Re/dt)*inner(u,v)*dx+0.5*inner(grad(u),grad(v))*dx
L5=(Re/dt)*inner(us,v)*dx+0.5*(inner(p1,div(v))*dx-inner(p0,div(v))*dx)

# Temperature Update
a6 = inner(T,q)*dx + dt*kappa*alpha*inner(grad(T),grad(q))*dx + dt*inner(dot(u1,grad(T)),q)*dx
L6 = inner(T0,q)*dx + dt*alpha*(heatt*inner(T0,q)*ds(1) + inner(mu*gamdot,q)*dx)  #+ inner(,q)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation


# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A5 = assemble(a5)
A6 = assemble(a6)


# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Time-stepping
t = dt
iter = 0            # iteration counter
maxiter = 500
while t < Tf + DOLFIN_EPS and iter < maxiter:
    
    print("t =", t)
    print("iteration", iter)
    iter += 1
    # Compute tentative velocity step
    #begin("Computing tentative velocity")
    A1 = assemble(a1)
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u12.vector(), b1, "bicgstab", "default")
    end()
    
    print(norm(u12.vector(),'linf'))
    
    #Compute Predicted U* Equation
    A2 = assemble(a2)
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcu]
    solve(A2, us.vector(), b2, "bicgstab", "default")
    end()
    #print(norm(us.vector(),'linf'))
    
    #Continuity Equation 1
    A3 = assemble(a3)
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcp]
    #[bc.apply(p1.vector()) for bc in bcp]
    solve(A3, p1.vector(), b3, "bicgstab", "default")
    end()
  
    h= p1.vector()-p0.vector()

    print 'Pressure Difference:', norm(h,'linf')

    #Velocity Update
    A5 = assemble(a5)
    b5 = assemble(L5)
    [bc.apply(A5, b5) for bc in bcu]
    solve(A5, u1.vector(), b5, "bicgstab", "default")
    end()

    m=u1.vector()-u0.vector()

    
    """#Temperature Equation
    A6 = assemble(a6)
    b6 = assemble(L6)
    [bc.apply(A6, b6) for bc in bcT]
    solve(A6, T1.vector(), b6, "bicgstab", "default")
    end()"""

    #Update Velocity Boundary Condition
    w.t = t
    pp.t = t
    sl.t = t

    # Plot solution
    plot(p1, title="Pressure", rescale=True)
    plot(u1, title="Velocity", rescale=True, mode = "auto")
    #plot(T1, title="Temperature", rescale=True)

    # Move to next time step
    u0.assign(u1)
    T0.assign(T1)
    p00.assign(p0)
    p0.assign(p1)
    t += dt


# Matlab Plot of the Solution
mplot(p1)
plt.show()


#Plot Contours USING MATPLOTLIB
# Scalar Function code


x = Expression('x[0]')  #GET X-COORDINATES LIST
y = Expression('x[1]')  #GET Y-COORDINATES LIST
pvals = p1.vector().array() # GET SOLUTION u= u(x,y) list
xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
xvalsv = interpolate(x, Q)#xyvals[:,0]
yvalsv= interpolate(y, Q)#xyvals[:,1]

xvals = xvalsv.vector().array()
yvals = yvalsv.vector().array()


xx = np.linspace(x0,x1)
yy = np.linspace(y0,y1)
XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

plt.contour(XX, YY, pp)
plt.show()


#Plot Contours USING MATPLOTLIB
# Vector Function code

g=list()
h=list()
n= mesh.num_vertices()
print(u1.vector().array())   # u is the FEM SOLUTION VECTOR IN FUNCTION SPACE 
for i in range(n):
    g.append(u1.vector().array()[2*i+1])
    h.append(u1.vector().array()[2*i])

uvals = np.asarray(h) # GET SOLUTION (u,v) -> u= u(x,y) list
vvals = np.asarray(g) # GET SOLUTION (u,v) -> v= v(x,y) list


xy = Expression(('x[0]','x[1]'))  #GET MESH COORDINATES LIST
xyvalsv = interpolate(xy, V)

q=list()
r=list()

#for i in range(n):
   #q.append(xyvalsv.vector().array()[2*i+1])
   # r.append(xyvalsv.vector().array()[2*i])


plot(mesh, interactive=True)


XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()

uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 

plot3 = plt.figure()
plt.streamplot(XX, YY, uu, vv, density=[0.5, 1], color='DarkRed', linewidth=0.5)       # STREAMLINE PLOT

plt.title('Stream Plot, Dynamic Colour')
plt.show(plot3)                     # display the plot


# Hold plot
plot(mesh, interactive=True)



