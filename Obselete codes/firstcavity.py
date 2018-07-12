"""Lid Driven Cavity Problem for an INCOMPRESSIBLE Oldroyd-B Fluid"""
"""Solution Method: Finite Element Method using DOLFIN (FEniCS)"""

"""INCOMPRESSIBLE Euler METHOD"""


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
m=40
nx=m*B
ny=m*L

c = min(x1-x0,y1-y0)

mesh= RectangleMesh(Point(x0,y0), Point(x1, y1), nx, ny)

print('Number of Cells:', mesh.num_cells())
print('Number of Vertices:', mesh.num_vertices())


# Mesh Refine Code

for i in range(2):
      g = (max(x1,y1)-max(x0,y0))*0.1/(5*i+1)
      cell_domains = CellFunction("bool", mesh)
      cell_domains.set_all(False)
      for cell in cells(mesh):
          x = cell.midpoint()
          if  (x[0] < x0+g and x[1] > y1-g) or (x[0] > x1-g and x[1] > y1-g): # or (x[0] < x0+g and x[1] < y0+g)  or (x[0] > x1-g and x[1] < g): 
              cell_domains[cell]=True
      #plot(cell_domains, interactive=True)
      mesh = refine(mesh, cell_domains, redistribute=True)
      print('Number of Cells:', mesh.num_cells())
      print('Number of Vertices:', mesh.num_vertices())

#Define Boundaries 


class Omega0(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] < DOLFIN_EPS and on_boundary else False 
                                                                          
class Omega1(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] > L*(1.0 - DOLFIN_EPS) and on_boundary  else False   


class Omega2(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] > B*(1.0 - DOLFIN_EPS)  and on_boundary else False 


class Omega3(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] < DOLFIN_EPS and on_boundary else False   

omega1= Omega1()
omega0= Omega0()
omega2= Omega2()
omega3= Omega3()


# MARK SUBDOMAINS (Create mesh functions over the cell facets)
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(5)
omega0.mark(sub_domains, 0)
omega1.mark(sub_domains, 2)
omega2.mark(sub_domains, 3)
omega3.mark(sub_domains, 4)

plot(sub_domains, interactive=True)

# Define function spaces (P2-P1)
d=2

V = VectorFunctionSpace(mesh, "CG", d)
Q = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", d)

# Define trial and test functions
u = TrialFunction(V)
rho=TrialFunction(Q)
p = TrialFunction(Q)
T = TrialFunction(W)
v = TestFunction(V)
q = TestFunction(Q)
r = TestFunction(W)


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
mu=Function(W)
T0=Function(W)
T1=Function(W)


boundary_parts = FacetFunction("size_t", mesh)
omega0.mark(boundary_parts,0)
omega1.mark(boundary_parts,1)
omega2.mark(boundary_parts,2)
omega3.mark(boundary_parts,3)
ds = Measure("ds")[boundary_parts]


# Set parameter values
h = mesh.hmin()
#print(h)
dt = 0.0005  #Time Stepping   
Tf = 3.5    #Final Time
cp = 1000.0
U = 1
Uv=Expression(('U*0.5*(1+tanh(6*t-5))','0'), t=0.0 ,U=U, d=d, degree=d)
mu_0 = 50.0*10E-1
Rc = 3.33*10E1
T_0 = 300.0 
T_h = 350.0      #Reference temperature
C=250.0 #Sutherland's Constant
kappa = 2.0
heatt= 0.00
rho=100.0
Pr=20.0 #Prandtl Number
Ra=60.0  #Rayleigh Number
V_h=0.01   #Viscous Heating Number
kappa = 20.0
heatt= 0.1
beta = 69*10E-2               # Thermal Expansion Coefficient
alpha=1.0/(rho0*cp)

k = Constant(dt)

# Nondimensional Parameters

 # Non Thermal
Re = rho*U*c/mu_0                                #Reynolds Number

 # Thermal
#Ra_G = rho*g*beta*c*c*c*(T_h-T_0)/(mu_0*alpha)         #Rayleigh Number (GASES)
#Ra_L = rho*g*c*c*c/(mu_0*alpha*(1+beta*(T_h-T_0)))     #Rayleigh Number (Liquids)
#Pr = 1#mu_0/(rho_0*alpha)                                  # Prandtl Number 
#Vh = (mu_0*alpha)/(cp*(c**2))                            # Viscous Heating Number 

print'Reynolds Number:',Re



# Define boundary conditions
td= Constant('5')
e = Constant('6')
w=Expression('T_0+0.5*(1.0+tanh(e*t-2.5))*(T_h-T_0)', t=0.0, e=e, T_0=T_0, T_h=T_h)
sl=Expression(('0.5*(1+tanh(e*t-2.5))*25.0','0.0'), t=0.0, e=e)
pp=Expression('0.5*(1.0+tanh(e*t-2.5))*000.0', t=0.0, e=e)
uin=Expression(('-4*(x[1]-y0)*(x[1]-y1)', '0.0'), y0=y0, y1=y1)
uout=Expression(('-(x[1]-y0)*(x[1]-y1)+5', '0.0'), y0=y0, y1=y1)
T_bl = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/L)', T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
T_bb = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/B)', T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)

 # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
inflow = DirichletBC(V, uin, omega0)
outflow = DirichletBC(V, Uv, omega1)
noslip0  = DirichletBC(V, (0.0, 0.0), omega0)  # No Slip boundary conditions on the left wall
drive1  =  DirichletBC(V, Uv, omega1)  # No Slip boundary conditions on the upper wall
noslip2  = DirichletBC(V, (0.0, 0.0), omega2)  # No Slip boundary conditions on the right part of the flow wall
noslip3  = DirichletBC(V, (0.0, 0.0), omega3)  # No Slip boundary conditions on the left part of the flow wall
slip  = DirichletBC(V, sl, omega0)  # Slip boundary conditions on the second part of the flow wall 
temp0 =  DirichletBC(W, T_0, omega0)    #Temperature on Omega0 
temp2 =  DirichletBC(W, T_0, omega2)    #Temperature on Omega2 
temp3 =  DirichletBC(W, T_0, omega3)    #Temperature on Omega3 
press1 = DirichletBC(Q, pp, omega0)
press2 = DirichletBC(Q, 0.0, omega1)

#Collect Boundary Conditions
bcu = [noslip0, drive1, noslip2, noslip3]
bcp = []
bcT = [temp0, temp2]
bctau = []


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



#Define Variable Parameters, Strain Rate and other tensors
sr = 0.5*(grad(u0) + transpose(grad(u0)))
gamdot = inner(sr,grad(u0))
thetal = (T)/(T_h-T_0)
thetar = (T_0)/(T_h-T_0)
thetar = project(thetar,W)
theta0 = (T0-T_0)/(T_h-T_0)
alpha = 1.0/(rho0*cp)

# TAYLOR GALERKIN METHOD

# Weak Formulation 

#Predicted U* Equation
a2=(Re/dt)*inner(u,v)*dx+0.5*inner(grad(u),grad(v))*dx
L2=(Re/dt)*inner(u0,v)*dx-0.5*inner(grad(u0),grad(v))*dx-Re*inner(grad(u12)*u12,v)*dx+inner(p0,div(v))*dx

#Continuity Equation 1
a3=inner(grad(p),grad(q))*dx
L3=inner(grad(p0),grad(q))*dx+(Re/dt)*inner(us,grad(q))*dx

#Velocity Update
a5=(Re/dt)*inner(u,v)*dx#+0.5*inner(grad(u),grad(v))*dx
L5=(Re/dt)*inner(us,v)*dx+0.5*(inner(p1,div(v))*dx-inner(p0,div(v))*dx)

# Temperature Update
#a6 = inner(T,q)*dx + k*kappa*alpha*inner(grad(T),grad(q))*dx + k*inner(inner(u1,grad(T)),q)*dx
#L6 = inner(T0,q)*dx + k*alpha*(heatt*(inner(grad(T0),n1*q)*ds(1)) + mu1*inner(gamdots,q)*dx + inner(gamdotp,q)*dx) #Neumann Condition on the outer Bearing is encoded in the weak formulation
      #+ inner(,r)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation


# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A5 = assemble(a5)
#A6 = assemble(a6)


# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Time-stepping
t = dt
iter = 0            # iteration counter
maxiter = 400
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

    
    #Temperature Equation
    #A6 = assemble(a6)
    #b6 = assemble(L6)
    #[bc.apply(A6, b6) for bc in bcT]
    #solve(A6, T1.vector(), b6, "bicgstab", "default")
    #end()

    # Plot solution
    plot(p1, title="Pressure", rescale=True)
    #plot(u1, title="Velocity", rescale=True, mode = "auto")
    #plot(T1, title="Temperature", rescale=True)

    # Move to next time step
    u0.assign(u1)
    T0.assign(T1)
    p00.assign(p0)
    p0.assign(p1)
    Uv.t=t
    t += dt

plot(u1, title="Velocity", rescale=True, mode = "auto")
plot(p1, title="Temperature", rescale=True)
interactive()

# Matlab Plot of the Solution
mplot(p1)
plt.show()
mplot(T1)
plt.show()

#Plot Contours USING MATPLOTLIB
# Scalar Function code


x = Expression('x[0]')  #GET X-COORDINATES LIST
y = Expression('x[1]')  #GET Y-COORDINATES LIST
pvals = p1.vector().array() # GET SOLUTION p= p(x,y) list
Tvals = T1.vector().array() # GET SOLUTION p= p(x,y) list
xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
xvalsq = interpolate(x, Q)#xyvals[:,0]
yvalsq= interpolate(y, Q)#xyvals[:,1]
xvalsw = interpolate(x, W)#xyvals[:,0]
yvalsw= interpolate(y, W)#xyvals[:,1]

xvals = xvalsq.vector().array()
yvals = yvalsq.vector().array()


xx = np.linspace(x0,x1)
yy = np.linspace(y0,y1)
XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

plt.contour(XX, YY, pp)
plt.show()

xvals = xvalsw.vector().array()
yvals = yvalsw.vector().array()

#TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='nn') #Construct Temperature Data
#plt.contour(XX, YY, TT) #Plot Temperature Data

#plt.show()



#Plot Contours USING MATPLOTLIB
# Vector Function code

g=list()
h=list()
n= mesh.num_vertices()
print(u1.vector().array())   # u is the FEM SOLUTION VECTOR IN FUNCTION SPACE 
for i in range(len(u1.vector().array())/2-1):
    g.append(u1.vector().array()[2*i+1])
    h.append(u1.vector().array()[2*i])

uvals = np.asarray(h) # GET SOLUTION (u,v) -> u= u(x,y) list
vvals = np.asarray(g) # GET SOLUTION (u,v) -> v= v(x,y) list


xy = Expression(('x[0]','x[1]'), d=d, degree=d)  #GET MESH COORDINATES LIST
xyvalsv = interpolate(xy, V)

q=list()
r=list()

for i in range(len(u1.vector().array())/2-1):
   q.append(xyvalsv.vector().array()[2*i+1])
   r.append(xyvalsv.vector().array()[2*i])

xvals = np.asarray(r)
yvals = np.asarray(q)

# Interpoltate velocity field data onto matlab grid
uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 

#Determine Speed 
speed = np.sqrt(uu*uu+ vv*vv)

plot3 = plt.figure()
plt.streamplot(XX, YY, uu, vv,  
               density=3,              
               color=speed/speed.max(),  
               cmap=cm.gnuplot,                         # colour map
               linewidth=2.5*speed/speed.max())       # line thickness
                              # arrow size

#plt.colorbar()                      # add colour bar on the right

plt.title('Convective Heat Flow')

plt.show(plot3)                                                                      # display the plot

mplot(mesh)






