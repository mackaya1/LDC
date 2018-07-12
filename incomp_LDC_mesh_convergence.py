"""Inompressible Lid Driven Cavity Problem for an COMPRESSIBLE Oldroyd-B Fluid"""
"""Solution Method: Finite Element Method using DOLFIN (FEniCS)"""


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

# Skew Mapping EXPONENTIAL
N=4.0
def expskewcavity(x,y):
    xi = 0.5*(1+np.tanh(2*N*(x-0.5)))
    ups= 0.5*(1+np.tanh(2*N*(y-0.5)))
    return (xi,ups)

pi=3.14159265359
def skewcavity(x,y):
    xi = 0.5*(1-np.cos(x*pi))
    ups =0.5*(1-np.cos(y*pi))
    return (xi,ups)

def xskewcavity(x,y):
    xi = 0.5*(1-np.cos(x*pi))
    ups = y
    return (xi,ups)

def yskewcavity(x,y):
    xi = x
    ups = (np.cos(0.5*y*pi))
    return (xi,ups)

B=1     # Characteristic Length
L=1
# Define Geometry
B=1
L=1
x_0 = 0
y_0 = 0
x_1 = B
y_1 = L
u_rect = Rectangle(Point(0.0,0.0),Point(1.0,1.0))
#n_y=mm*L

#base_mesh= RectangleMesh(Point(x_0,y_0), Point(x_1, y_1), n_x, n_y) # Rectangular Mesh

def LDC_Mesh(mm):

    # Define Geometry
    B=1
    L=1
    x0 = 0
    y0 = 0
    x1 = B
    y1 = L

    # Mesh refinement comparison Loop

     
    nx=mm*B
    ny=mm*L

    c = min(x1-x0,y1-y0)
    base_mesh= RectangleMesh(Point(x0,y0), Point(x1, y1), nx, ny) # Rectangular Mesh


    # Create Unstructured mesh

    u_rec=Rectangle(Point(0.0,0.0),Point(1.0,1.0))
    mesh0=generate_mesh(u_rec, mm)



    #SKEW MESH FUNCTION

    # MESH CONSTRUCTION CODE

    nv= base_mesh.num_vertices()
    nc= base_mesh.num_cells()
    coorX = base_mesh.coordinates()[:,0]
    coorY = base_mesh.coordinates()[:,1]
    cells0 = base_mesh.cells()[:,0]
    cells1 = base_mesh.cells()[:,1]
    cells2 = base_mesh.cells()[:,2]



    # OLD MESH COORDINATES -> NEW MESH COORDINATES
    r=list()
    l=list()
    for i in range(nv):
        r.append(yskewcavity(coorX[i], coorY[i])[0])
        l.append(yskewcavity(coorX[i], coorY[i])[1])

    r=np.asarray(r)
    l=np.asarray(l)

    # MESH GENERATION (Using Mesheditor)
    mesh1 = Mesh()
    editor = MeshEditor()
    editor.open(mesh1, "triangle", 2,2)
    editor.init_vertices(nv)
    editor.init_cells(nc)
    for i in range(nv):
        editor.add_vertex(i, r[i], l[i])
    for i in range(nc):
        editor.add_cell(i, cells0[i], cells1[i], cells2[i])
    editor.close()

    return mesh1

# Mesh Refine Code (UNSTRUCTURED MESH)

def refine_boundary(mesh, times):
    for i in range(times):
          g = (max(x_1,y_1)-max(x_0,y_0))*0.025
          cell_domains = MeshFunction("bool", mesh, 2)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  (x[0] < x_0+g or x[1] < y_0+g) or (x[0] > x_1-g or x[1] > y_1-g): 
                  cell_domains[cell]=True
          mesh_refine = refine(mesh, cell_domains, redistribute=True)
    return mesh_refine

def refine_top(mesh, times):
    for i in range(times):
          g = (max(x_1,y_1)-max(x_0,y_0))*0.05#/(i+1)
          cell_domains = MeshFunction("bool", mesh, 2)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  x[1] > y_1-g:
                  cell_domains[cell]=True
          mesh_refine = refine(mesh, cell_domains, redistribute=True)
    return mesh_refine


# Adaptive Mesh Refinement 
def adaptive_refinement(mesh, kapp, ratio):
    kapp_array = kapp.vector().get_local()
    kapp_level = np.percentile(kapp_array, (1-ratio)*100)

    cell_domains = MeshFunction("bool", mesh, 2)
    cell_domains.set_all(False)
    for cell in cells(mesh):
        x = cell.midpoint()
        if  kapp([x[0], x[1]]) > kapp_level:
            cell_domains[cell]=True

    mesh = refine(mesh, cell_domains, redistribute=True)
    return mesh
# Some Useful Functions
def  tgrad (w):
    """ Returns  transpose  gradient """
    return  transpose(grad(w))
def Dincomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return (grad(w) + tgrad(w))/2
def Dcomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*Identity(len(u)))/2

def sigma(u, p, Tau):
    return 2*betav*Dincomp(u) - p*Identity(len(u)) + ((1-betav)/We)*(Tau-Identity(len(u)))

def sigmacom(u, p, Tau):
    return 2*betav*Dcomp(u) - p*Identity(len(u)) + ((1-betav)/We)*(Tau-Identity(len(u)))

def Fdef(u, Tau):
    return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u))

def Fdefcom(u, Tau):
    return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u)) + div(u)*Tau 

def normalize_solution(u):
    "Normalize u: return u divided by max(u)"
    u_array = u.vector().get_local()
    u_max = np.max(np.abs(u_array))
    u_array /= u_max
    u.vector()[:] = u_array
    #u.vector().set_local(u_array)  # alternative
    return u


def magnitude(u):
    return np.power((u[0]*u[0]+u[1]*u[1]), 0.5)

def euc_norm(tau):
    return np.power((tau[0]*tau[0] + tau[1]*tau[1] + tau[2]*tau[2]), 0.5)


def stream_function(u):
    '''Compute stream function of given 2-d velocity vector.'''
    V = u.function_space().sub(0).collapse()

    if V.mesh().topology().dim() != 2:
        raise ValueError("Only stream function in 2D can be computed.")

    psi = TrialFunction(V)
    phi = TestFunction(V)

    a = inner(grad(psi), grad(phi))*dx
    L = inner(u[1].dx(0) - u[0].dx(1), phi)*dx
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    A, b = assemble_system(a, L, bc)
    psi = Function(V)
    solve(A, psi.vector(), b)

    return psi

def comp_stream_function(rho, u):
    '''Compute stream function of given 2-d velocity vector.'''
    V = u.function_space().sub(0).collapse()

    if V.mesh().topology().dim() != 2:
        raise ValueError("Only stream function in 2D can be computed.")

    psi = TrialFunction(V)
    phi = TestFunction(V)

    a = inner(grad(psi), grad(phi))*dx
    L = inner(rho*u[1].dx(0) - rho*u[0].dx(1), phi)*dx
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    A, b = assemble_system(a, L, bc)
    psi = Function(V)
    solve(A, psi.vector(), b)

    return psi


def min_location(u):

    V = u.function_space()

    if V.mesh().topology().dim() != 2:
       raise ValueError("Only minimum of scalar function in 2D can be computed.")

    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))

    function_array = u.vector().get_local()
    minimum = min(u.vector().get_local())

    min_index = np.where(function_array == minimum)
    min_loc = dofs_x[min_index]

    return min_loc

def absolute(u):
    u_array = np.absolute(u.vector().get_local())
    u.vector()[:] = u_array
    return u

def l2norm_solution(u):
    u_array = u.vector().get_local()
    u_l2 = norm(u, 'L2')
    u_array /= u_l2
    u.vector()[:] = u_array
    return u

"""def zero_u(u):
    u_array = u.vector().get_local()
    for i in len(u_array):
        if u.vector().get_local()[i] < 10E-10:"""
           
    

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# FEM Solution Convergence/Energy Plot
x1=list()
x2=list()
x3=list()
x4=list()
x5=list()
x6=list()
y=list()
z=list()
zz=list()
zzz=list()
zl=list()
ek1=list()
ek2=list()
ek3=list()
ek4=list()
ee1=list()
ee2=list()
ee3=list()
ee4=list()
ek5=list()
ee5=list()
ek6=list()
ee6=list()
x_axis=list()
y_axis=list()
u_xg = list()
u_yg = list()
sig_xxg = list()
sig_xyg = list()
sig_yyg = list()

# Experiment Run Time
T_f = 4.0
Tf=T_f 
tol = 0.0001

# Default Nondimensional Parameters
conv = 0
U=1
betav = 0.5     
Re = 1                             #Reynolds Number
We = 0.2                          #Weisenberg NUmber
Di = 0.005                         #Diffusion Number
Vh = 0.005
T_0 = 300
T_h = 350

c0 = 1500
Ma = 0.001


c1 = 0.1
c2 = 0.001
th = 1.0               # DEVSS
#c1 = alph*h_ska        # SUPG / SU

# Loop Experiments
loopend = 4
j = 0
jj = 0
jjj = 0
err_count = 0
conv_fail = 0
jmesh=0
# Mesh Convergence Loop
while jmesh < loopend:
    j+=1    
    t=0.0
  
    jj=1
    if err_count == 0:
        jmesh+=1
        if jmesh==1:
            mm=36
            mesh = generate_mesh(u_rect, mm)
        if jmesh==2:
            mm=48
            mesh = generate_mesh(u_rect, mm)
        if jmesh==3:
            mm=50
            mesh = generate_mesh(u_rect, mm)
        if jmesh==4:
            mm=52 #(80)
            mesh = generate_mesh(u_rect, mm)


    dt = 2*mesh.hmin()**2

    gdim = mesh.geometry().dim() # Mesh Geometry

    mplot(mesh)
    plt.savefig("unstructured_mesh-"+str(mm)+".png")
    plt.clf()
    plt.close()  
    #quit()  

    #Define Boundaries 

    top_bound = 0.5*(1+tanh(N)) 

    class No_slip(SubDomain):
          def inside(self, x, on_boundary):
              return True if on_boundary else False 
                                                                              
    class Lid(SubDomain):
          def inside(self, x, on_boundary):
              return True if x[1] > L*(top_bound - DOLFIN_EPS) and on_boundary  else False   

    no_slip = No_slip()
    lid = Lid()


    # MARK SUBDOMAINS (Create mesh functions over the cell facets)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(5)
    no_slip.mark(sub_domains, 0)
    lid.mark(sub_domains, 2)


    #file = File("subdomains.pvd")
    #file << sub_domains



    #Define Boundary Parts
    boundary_parts = FacetFunction("size_t", mesh)
    no_slip.mark(boundary_parts,0)
    lid.mark(boundary_parts,1)
    ds = Measure("ds")[boundary_parts]

    # Define function spaces (P2-P1)

    # Discretization  parameters
    family = "CG"; dfamily = "DG"; rich = "Bubble"
    shape = "triangle"; order = 2

    #mesh.ufl_cell()

    V_s = VectorElement(family, mesh.ufl_cell(), order)       # Elements
    Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)
    Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
    Z_d = VectorElement(dfamily, mesh.ufl_cell(),  order-2, 3)
    Z_se = VectorElement(rich, mesh.ufl_cell(),  order+1, 3)
    Q_s = FiniteElement(family, mesh.ufl_cell(), order-1)
    Q_p = FiniteElement(rich, mesh.ufl_cell(), order+1, 3)
    Z_e = EnrichedElement(Z_c,Z_se)                 # Enriched Elements
    Q_rich = EnrichedElement(Q_s,Q_p)


    W = FunctionSpace(mesh,V_s*Z_s)             # F.E. Spaces 
    V = FunctionSpace(mesh,V_s)

    Z = FunctionSpace(mesh,Z_s)
    Zd = FunctionSpace(mesh,Z_d)
    #Ze = FunctionSpace(mesh,Z_e)
    Zc = FunctionSpace(mesh,Z_c)
    Q = FunctionSpace(mesh,Q_s)
    Qt = FunctionSpace(mesh, "DG", order-2)
    Qr = FunctionSpace(mesh,Q_s)


    # Define trial and test functions [TAYLOR GALERKIN Method]
    rho=TrialFunction(Q)
    p = TrialFunction(Q)
    T = TrialFunction(Q)
    q = TestFunction(Q)
    r = TestFunction(Q)

    p0=Function(Q)       # Pressure Field t=t^n
    p1=Function(Q)       # Pressure Field t=t^n+1
    rho0=Function(Q)
    rho1=Function(Q)
    T0=Function(Q)       # Temperature Field t=t^n
    T1=Function(Q)       # Temperature Field t=t^n+1


    (v, R_vec) = TestFunctions(W)
    (u, D_vec) = TrialFunctions(W)

    tau_vec = TrialFunction(Zc)
    Rt_vec = TestFunction(Zc)


    tau0_vec=Function(Zc)     # Stress Field (Vector) t=t^n
    tau12_vec=Function(Zc)    # Stress Field (Vector) t=t^n+1/2
    tau1_vec=Function(Zc)     # Stress Field (Vector) t=t^n+1

    w0= Function(W)
    w12= Function(W)
    ws= Function(W)
    w1= Function(W)

    (u0, D0_vec) = w0.split()
    (u12, D12_vec) = w12.split()
    (u1, D1_vec) = w1.split()
    (us, Ds_vec) = ws.split()


    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), degree=2)


    # The  projected  rate -of-strain
    D_proj_vec = Function(Z)
    D_proj = as_matrix([[D_proj_vec[0], D_proj_vec[1]],
                        [D_proj_vec[1], D_proj_vec[2]]])

    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), degree=2)


    # Initial Conformation Tensor
    I_vec = Expression(('1.0','0.0','1.0'), degree=2)
    initial_guess_conform = project(I_vec, Zc)



    # Project Vector Trial Functions of Stress onto SYMMETRIC Tensor Space

    D =  as_matrix([[D_vec[0], D_vec[1]],
                    [D_vec[1], D_vec[2]]])

    tau = as_matrix([[tau_vec[0], tau_vec[1]],
                     [tau_vec[1], tau_vec[2]]])  

    # Project Vector Test Functions of Stress onto SYMMETRIC Tensor Space

    Rt = as_matrix([[Rt_vec[0], Rt_vec[1]],
                     [Rt_vec[1], Rt_vec[2]]])        # DEVSS Space

    R = as_matrix([[R_vec[0], R_vec[1]],
                     [R_vec[1], R_vec[2]]])

    # Project Vector Functions of Stress onto SYMMETRIC Tensor Space

    D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                    [D0_vec[1], D0_vec[2]]])        #DEVSS STABILISATION

    D12 = as_matrix([[D12_vec[0], D12_vec[1]],
                    [D12_vec[1], D12_vec[2]]])

    Ds = as_matrix([[Ds_vec[0], Ds_vec[1]],
                    [Ds_vec[1], Ds_vec[2]]])


    D1 = as_matrix([[D1_vec[0], D1_vec[1]],
                    [D1_vec[1], D1_vec[2]]]) 


    tau0 = as_matrix([[tau0_vec[0], tau0_vec[1]],
                      [tau0_vec[1], tau0_vec[2]]])        # Stress 

    tau12 = as_matrix([[tau12_vec[0], tau12_vec[1]],
                       [tau12_vec[1], tau12_vec[2]]]) 

    tau1 = as_matrix([[tau1_vec[0], tau1_vec[1]],
                      [tau1_vec[1], tau1_vec[2]]])   


    # Initial Conformation Tensor
    I_vec = Expression(('1.0','0.0','1.0'), degree=2)
    initial_guess_conform = project(I_vec, Zc)
    assign(tau0_vec, initial_guess_conform)         # Initial guess for conformation tensor is Identity matrix   


    # Define boundary/stabilisation FUNCTIONS

    ulidreg=Expression(('8.*(1.0+tanh(8*t-4.0))*(x[0]*(L-x[0]))*(x[0]*(L-x[0]))','0'), degree=2, t=0.0, L=L, e=e, T_0=T_0, T_h=T_h) # Lid Speed 
    ulidreg_test=Expression(('16.*(x[0]*(L-x[0]))*(x[0]*(L-x[0]))','0'), degree=2, t=0.0, L=L, e=e, T_0=T_0, T_h=T_h) # Lid Speed 
    ulid=Expression(('0.5*(1.0+tanh(8*t-4.0))','0'), degree=2, t=0.0, T_0=T_0, T_h=T_h) # Lid Speed 
    T_bl = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/L)', degree=2, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
    T_bb = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/B)', degree=2, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
    h_sk = Expression('cos(pi*x[0])-cos(pi*(x[0]+1/mm))','cos(pi*x[1])-cos(pi*(x[1]+1/mm))', degree=2, pi=pi, mm=mm, L=L, B=B)             # Mesh size function
    h_k = Expression(('1/mm','1/mm'), degree=2, mm=mm, L=L, B=B)
    h_m = Expression('0.5*h', degree=2, h=mesh.hmin())
    h_ka = Expression('0.5*1/mm', degree=2, mm=mm, L=L, B=B)
    h_ska= Expression('cos(0.5*pi*(1.0-(x[1]+1/mm)))-cos(0.5*pi*(1.0-x[1]))', degree=2, pi=pi, mm=mm, L=L, B=B)
    rampd=Expression('0.5*(1+tanh(8*(2.0-t)))', degree=2, t=0.0)
    rampu=Expression('0.5*(1+tanh(16*(t-2.0)))', degree=2, t=0.0)

    # Set Boundary Function Time = 0
    rampd.t=t
    ulid.t=t
    ulidreg.t=t



    # Mesh Related Parameters Functions
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

    h_ska = project(h_ska, Qt)
    #mplot(h_ska)
    #plt.colorbar()
    #plt.savefig("mesh_parameter.png")

    #plt.clf()
    #plt.close()
    #quit()

    # Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
    n0 =  Expression(('-1' , '0'), degree=2)
    n1 =  Expression(('0' , '1' ), degree=2)
    n2 =  Expression(('1' , '0' ), degree=2)
    n3 =  Expression(('0' , '-1'), degree=2)

    # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
    noslip  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), no_slip)  # No Slip boundary conditions on the left wall
    if jj==0:
        drive  =  DirichletBC(W.sub(0), ulidreg_test, lid)  # No Slip boundary conditions on the upper wall
    if jj==1:
        drive  =  DirichletBC(W.sub(0), ulidreg, lid)  # No Slip boundary conditions on the upper wall
    #slip  = DirichletBC(V, sl, omega0)  # Slip boundary conditions on the second part of the flow wall 
    #temp0 =  DirichletBC(Qt, T_0, omega0)    #Temperature on Omega0 
    #temp2 =  DirichletBC(Qt, T_0, omega2)    #Temperature on Omega2 
    #temp3 =  DirichletBC(Qt, T_0, omega3)    #Temperature on Omega3 
    #Collect Boundary Conditions
    bcu = [noslip, drive]
    bcp = []
    bcT = [] # temp0, temp2
    bctau = []



    # Set Stabilisation Parameters


    # Comparing different WEISSENBERG Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=__
    """conv=0                                    # Non-inertial Flow Parameter (Re=0)
    Re=1.0
    if j==1:
       We=0.1
    elif j==2:
       We=0.25
    elif j==3:
       We=0.5
    elif j==4:
       We=0.75
    elif j==5:
       We=1.0"""


    # Comparing different REYNOLDS NUMBERS Numbers (Re=0,5,10,25,50) at We=0.5
    """conv=1                                      # Non-inertial Flow Parameter (Re=0)
    We=0.4
    if j==1:
       conv=10E-8
       Re=1
    elif j==2:
       Re=5
    elif j==3:
       Re=10
    elif j==4:
       Re=25
    elif j==5:
       Re=50"""


    # Comparing Effect of DEVSS/ SUPG Stabilisation Parameter
    """alph = 0.125
    th=10E-16
    c1=alph*h_ska    #SUPG Stabilisation
    We=0.5
    conv=10E-15
    Re=1
    if j==1:
        th=0
    elif j==2:
        th=0.1*(1.0-betav)
    elif j==3:
        th=0.5*(1.0-betav)"""



    # Comparing Effect of Diffusion Stabilisation Parameter
    """c1=h_ka     #SUPG Stabilisation
    th=0.1*(1.0-betav)          #DEVSS Stabilisation
    We=0.5
    conv=10E-15
    Re=1
    if j==1:
        c2=10E-6*h_ka
    elif j==2:
        c2=rampd*0.1*h_ka"""

    # Comparing the Effect of SUPG Stabilisation
    """th=10E-16        #DEVSS Stabilisation
    c2=10E-6*h_ka    #Diffusion Stabilisation
    We=0.5
    Re=10
    if j==1:
        c1=h_ka*10E-10
    elif j==2:
        c1=0.1*h_ka
    elif j==3:
        c1=h_ka"""

    # Adaptive Mesh Refinement Step
    if jj==0 and err_count < 2: # 0 = on, 1 = off
       We = 1.0
       betav = 0.5
       Tf = 1.5*(1 + 2*err_count*0.25)
       dt = 10*mesh.hmin()**2 
       th = 0.0


    
        

    # Continuation in Reynolds/Weissenberg Number Number (Re-->20Re/We-->20We)
    Ret=Expression('Re*(1.0+19.0*0.5*(1.0+tanh(0.7*t-4.0)))', t=0.0, Re=Re, degree=2)
    Rey=Re
    Wet=Expression('(We/100)*(1.0+99.0*0.5*(1.0+tanh(0.7*t-5.0)))', t=0.0, We=We, degree=2)
    if jj==0:
        print '############# ADAPTIVE MESH REFINEMENT STAGE ################'   
        print 'Number of Refinements:', err_count 
    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', L
    print 'Characteristic Velocity (m/s):', 1.0
    print 'Lid velocity:', (0.5*(1.0+tanh(e*t-3.0)),0)
    print 'Reynolds Number:', Rey
    print 'Non-inertial parameter:', conv
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh

    Np= len(p0.vector().get_local())
    Nv= len(w0.vector().get_local())   
    Ntau= len(tau0_vec.vector().get_local())
    dof= 3*Nv+2*Ntau+Np
    print '############# Discrete Space Characteristics ############'
    print 'Degree of Velocity Elements', order
    print 'Mesh: %s x %s' %(mm, mm)
    print('Size of Pressure Space = %d ' % Np)
    print('Size of Velocity Space = %d ' % Nv)
    print('Size of Stress Space = %d ' % Ntau)
    print('Degrees of Freedom = %d ' % dof)
    print 'Number of Cells:', mesh.num_cells()
    print 'Number of Vertices:', mesh.num_vertices()
    print 'Minimum Cell Diamter:', mesh.hmin()
    print 'Maximum Cell Diamter:', mesh.hmax()
    print '############# Stabilisation Parameters ############'
    print 'DEVSS Parameter:', th


    

     
    #Define Variable Parameters, Strain Rate and other tensors
    sr = (grad(u) + transpose(grad(u)))
    srg = grad(u)
    gamdots = inner(Dincomp(u1),grad(u1))
    gamdotp = inner(tau1,grad(u1))
    thetal = (T)/(T_h-T_0)
    thetar = (T_0)/(T_h-T_0)
    thetar = project(thetar,Qt)
    theta0 = (T0-T_0)/(T_h-T_0)

 
    """# STABILISATION TERMS
    F1 = dot(u1,grad(tau)) - dot(grad(u1),tau) - dot(tau,tgrad(u1))                                                # Convection/Deformation Terms
    F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12))                                            # Convection/Deformation Terms

    F1R = dot(u1,grad(tau1)) - dot(grad(u1),tau1) - dot(tau1,tgrad(u1))

    # SU/SUPG Stabilisation
 
    velocity = project(u1,V)  
    unorm = norm(velocity.vector(),'linf')
    #u_av = project((u1[0]**2+u1[1]**2),Q)
    h = CellSize(mesh)
    eta_x = u1[0]
    eta_y = u1[1]
    xi = (eta_x)
    ups = (eta_y)
    xi = tanh(xi)
    xi = project(xi,Qt)
    ups = tanh(ups) #ups - (1/6)*(ups**3)+(2/15)*(ups**5)-(17/315)*(ups**7)
    ups = project(ups,Qt)
    c1 = alph1*(h/(2.0))*(xi*u1[0]+ups*u1[1])
    c1 = project(c1,Qt)


    c2 = alph2*h
    c3 = alph3*h

    # SU Stabilisation
    SUl3 = inner(c1*dot(u0 , grad(Rt)), dot(u12, grad(tau)))*dx
    SUl4 = inner(c1*dot(u1, grad(Rt)), dot(u1, grad(tau)))*dx


    # SUPG Stabilisation
    res12 = tau + We*F12
    res1 = tau + We*F1 
    
    SUPGl3 = inner(tau+We*F12,c1*dot(u12,grad(Rt)))*dx
    SUPGr3 = inner(Dincomp(u12),c1*dot(u12,grad(Rt)))*dx    
    SUPGl4 = inner(res1,c1*dot(u1,grad(Rt)))*dx
    SUPGr4 = inner(Dincomp(u1),c1*dot(u1,grad(Rt)))*dx """

    # Update Stabilisation (Copy and Paste Stabilisation Technique from above)
    """F1R = Fdef(u1, tau1)  #Compute the residual in the STRESS EQUATION
    F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
    Dcomp1_vec = as_vector([Dcomp(u1)[0,0], Dcomp(u1)[1,0], Dcomp(u1)[1,1]])
    restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - I_vec
    res_test = project(restau0, Zd)
    res_orth = project(restau0-res_test, Zc)                                
    res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
    res_orth_norm = np.power(res_orth_norm_sq, 0.5)
    kapp = project(res_orth_norm, Q)
    LPSl_stress = inner(kapp*grad_tau_stab*grad(tau),grad(Rt))*dx """

    # Update Stabilisation (Copy and Paste Stabilisation Technique from above)
    """grad_tau_sq = inner(grad(tau1),grad(tau1))
    #grad_tau_sq = as_vector([grad_tau_sq[0,0], grad_tau_sq[1,0], grad_tau_sq[1,1]])
    grad_tau_proj = project(grad_tau_sq, Qt)
    grad_tau_orth = project(grad_tau_sq - grad_tau_proj, Q) 
    grad_tau_orth_norm = project(inner(grad_tau_orth, grad_tau_orth), Q)
    grad_tau_stab = np.power(grad_tau_orth_norm, 0.5)
                                   
    u_norm_sq = project(u1[0]*u1[0]+u1[1]*u1[1], Q)
    u_norm = np.power(u_norm_sq, 0.5)
    grad_u_norm_sq = project(inner(grad(u1),grad(u1)), Q)
    grad_u_norm = np.power(grad_u_norm_sq, 0.5)
    kapp = project(c1*h*u_norm + c2*h*h*grad_u_norm, Q)
    
    LPSl_stress = inner(kapp*grad_tau_stab*grad(tau),grad(Rt))*dx""" 

    #SUPGl = inner(We*F1R,c1*dot(u1,grad(Rt)))*dx


    # LPS Stabilisation

    #LPSl4 = inner(c2*div(tau),div(Rt))*dx + inner(c3*grad(tau),grad(Rt))*dx

    # DEVSS Stabilisation
    
    DEVSSl_u12 = 2*(1-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSr_u12 = 2*inner(D0,Dincomp(v))*dx   
    DEVSSl_u1 = 2*(1-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSr_u1 = 2*inner(D12,Dincomp(v))*dx   


    # Governing Equations - Taylor-Galerkin Scheme

    # VELOCITY HALF STEP
    (u0, D0_vec)=w0.split()  
    D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                    [D0_vec[1], D0_vec[2]]])                    #DEVSS STABILISATION
    U = 0.5*(u + u0)   

   
    # Set up Krylov Solver 

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True
    parameters['krylov_solver']['monitor_convergence'] = False
    
    solveru = KrylovSolver("bicgstab", "default")
    solvertau = KrylovSolver("bicgstab", "default")
    solverp = KrylovSolver("bicgstab", "default")

    #Folder To Save Plots for Paraview
    #fv=File("Velocity Results Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
 

    #Lists for Energy Values
    x=list()
    ee=list()
    ek=list()

    #ftau=File("Incompressible Viscoelastic Flow Results/Paraview/Stress_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/stress "+str(t)+".pvd")
    #fv=File("Incompressible Viscoelastic Flow Results/Paraview/Velocity_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/velocity "+str(t)+".pvd")

    # Time-stepping
    t = 0.0
    iter = 0            # iteration counter
    maxiter = 10000000
    if jj==0:
       maxiter = 15
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s - %s" %(t, iter, conv_fail, jmesh, j)
        #print udiff

        rampd.t=t
        ulid.t=t
        ulidreg.t=t
        Ret.t=t
        Wet.t=t

        if jj==1:
            # Update Stabilisation (Copy and Paste Stabilisation Technique from above)
            F1R = Fdef(u1, tau1)  #Compute the residual in the STRESS EQUATION
            F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
            Dincomp1_vec = as_vector([Dincomp(u1)[0,0], Dincomp(u1)[1,0], Dincomp(u1)[1,1]])
            restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec  - I_vec #- diss_vec 
            res_test = project(restau0, Zd)
            res_orth = project(restau0-res_test, Zc)                                
            res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
            res_orth_norm = np.power(res_orth_norm_sq, 0.5)
            kapp = project(res_orth_norm, Qt)
            kapp = absolute(kapp)
            LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation
        if jj==0:
            LPSl_stress = 0
        
        #LPSl_stress = inner(kapp*grad(tau),grad(Rt))*dx

        #print u_norm_sq.vector().get_local() 
        #print "     "
        #print kapp.vector().get_local().max() 
           
        # Update Solutions
        if iter > 1:
            w0.assign(w1)
            T0.assign(T1)
            rho0.assign(rho1)
            p0.assign(p1)
            tau0_vec.assign(tau1_vec)
             

        (u0, D0_vec) = w0.split()  

        D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                        [D0_vec[1], D0_vec[2]]])                    #DEVSS STABILISATION
        DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS
        U = 0.5*(u + u0)              

        # Velocity Half Step
        """lhsFu12 = Re*(2.0*(u - u0) / dt + conv*dot(u0, nabla_grad(u0)))
        Fu12 = dot(lhsFu12, v)*dx + \
               + inner(sigma(U, p0, tau0), Dincomp(v))*dx \
               + dot(p0*n, v)*ds - dot(betav*nabla_grad(U)*n, v)*ds\
               - ((1.0-betav)/We)*dot(tau0*n, v)*ds\
               + inner(D-Dincomp(u),R)*dx 
        a1 = lhs(Fu12)
        L1 = rhs(Fu12)
        a1_stab = a1 + th*DEVSSl_u12 
        L1_stab = L1 + th*DEVSSr_u12
        A1 = assemble(a1_stab)
        b1= assemble(L1_stab)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, w12.vector(), b1, "bicgstab", "default")
        end()

        (u12, D12_vec) = w12.split()
        D12 = as_matrix([[D12_vec[0], D12_vec[1]],
                        [D12_vec[1], D12_vec[2]]])"""


        DEVSSr_u1 = 2*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS

        """# STRESS Half Step
        F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12)) # Convection/Deformation Terms
        lhs_tau12 = (We/dt+1.0/We)*tau + F12                             # Left Hand Side
        rhs_tau12= (We/dt)*tau0 + (1/We)*I                     # Right Hand Side

        a3 = inner(lhs_tau12,Rt)*dx                                 # Weak Form
        L3 = inner(rhs_tau12,Rt)*dx

        a3 += SUPGl3             # SUPG Stabilisation LHS
        L3 += SUPGr3             # SUPG / SU Stabilisation RHS
        A3=assemble(a3)
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solve(A3, tau12_vec.vector(), b3, "bicgstab", "default")
        end()"""
        
        #Predictor Step [U*]
        lhsFus = Re*((u - u0)/dt + conv*dot(u0, nabla_grad(U)))
        Fus = dot(lhsFus, v)*dx + \
              + inner(sigma(U, p0, tau0), Dincomp(v))*dx \
              + 0.5*dot(p0*n, v)*ds - betav*(dot(nabla_grad(U)*n, v)*ds) \
              - ((1.0-betav)/We)*dot(tau0*n, v)*ds\
               + inner(D-Dincomp(u),R)*dx   
              
        a2= lhs(Fus)
        L2= rhs(Fus) 
        a2_stab = a2 + th*DEVSSl_u1 
        L2_stab = L2 + th*DEVSSr_u1
        A2 = assemble(a2_stab)
        b2 = assemble(L2_stab)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, ws.vector(), b2, "bicgstab", "default")
        end()
        (us, Ds_vec) = ws.split()

        # Pressure Correction
        a5=inner(grad(p),grad(q))*dx 
        L5=inner(grad(p0),grad(q))*dx + (Re/dt)*inner(us,grad(q))*dx
        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()
        
        #Velocity Update
        lhs_u1 = (Re/dt)*u                                          # Left Hand Side
        rhs_u1 = (Re/dt)*us                                         # Right Hand Side
        a7=inner(lhs_u1,v)*dx + inner(D-Dincomp(u),R)*dx                                           # Weak Form
        L7=inner(rhs_u1,v)*dx + 0.5*inner(p1-p0,div(v))*dx - 0.5*dot(p1*n, v)*ds
        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, w1.vector(), b7)
        end()
        (u1, D1_vec) = w1.split()
        D1 = as_matrix([[D1_vec[0], D1_vec[1]],
                        [D1_vec[1], D1_vec[2]]])

        U12 = 0.5*(u1 + u0)   

        # Stress Full Step
        stress_eq = (We/dt+1.0)*tau  +  We*Fdef(u1,tau) - (We/dt)*tau0 - Identity(len(u))
        A = inner(stress_eq,Rt)*dx
        a4 = lhs(A)
        L4 = rhs(A) 
        a4_stab = a4 + LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
        L4 += 0            # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   

        A4=assemble(a4_stab)                                     # Assemble System
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solvertau.solve(A4, tau1_vec.vector(), b4)
        end()

        #stress_vec = ((1-betav)/We)*(tau1_vec-I_vec)
        #stress = interpolate(stress_vec, Z)

        # Temperature Update (FIRST ORDER)
        #lhs_theta1 = (1.0/dt)*thetal + dot(u1,grad(thetal))
        #rhs_theta1 = (1.0/dt)*thetar + dot(u1,grad(thetar)) + (1.0/dt)*theta0 + Vh*gamdots
        #a8 = inner(lhs_theta1,r)*dx + Di*inner(grad(thetal),grad(r))*dx 
        #L8 = inner(rhs_theta1,r)*dx + Di*inner(grad(thetar),grad(r))*dx + Bi*inner(grad(theta0),n1*r)*ds(1) 

        # Energy Calculations
        E_k=assemble(0.5*dot(u1,u1)*dx)
        E_e=assemble((tau1_vec[0]+tau1_vec[2]-2.0)*dx)

        
        
        # Record Elastic & Kinetic Energy Values (Method 1)
        if j==1:
           x1.append(t)
           ek1.append(E_k)
           ee1.append(E_e)
        if j==2:
           x2.append(t)
           ek2.append(E_k)
           ee2.append(E_e)
        if j==3:
           x3.append(t)
           ek3.append(E_k)
           ee3.append(E_e)
        if j==4:
           x4.append(t)
           ek4.append(E_k)
           ee4.append(E_e)
        if j==5:
           x5.append(t)
           ek5.append(E_k)
           ee5.append(E_e)
        if j==6:
           x6.append(t)
           ek6.append(E_k)
           ee6.append(E_e)

        # Record Error Data 


        
        #shear_stress=project(tau1[1,0],Q)
        # Save Plot to Paraview Folder 
        #for i in range(5000):
        #    if iter== (0.01/dt)*i:
        #       ftau << shear_stress


        # Break Loop if code is diverging

        if norm(w1.vector(), 'linf') > 10E5 or np.isnan(sum(w1.vector().get_local())) or abs(E_k) > 10:
            print 'FE Solution Diverging'   #Print message 
            #with open("DEVSS Weissenberg Compressible Stability.txt", "a") as text_file:
                 #text_file.write("Iteration:"+str(j)+"--- Re="+str(Rey)+", We="+str(We)+", t="+str(t)+", dt="+str(dt)+'\n')
            if j==1:           # Clear Lists
               x1=list()
               ek1=list()
               ee1=list()
            if j==2:
               x2=list()
               ek2=list()
               ee2=list()
            if j==3:
               x3=list()
               ek3=list()
               ee3=list()
            if j==4:
               x4=list()
               ek4=list()
               ee4=list()
            if j==5:
               x5=list()
               ek5=list()
               ee5=list() 
            j-=1                            # Extend loop
            jj=0                         # Convergence Failures

            alph1 = 2*alph1
            if jj>5:
               Tf= (iter-10)*dt

            # Reset Functions
            p0=Function(Q)       # Pressure Field t=t^n
            p1=Function(Q)       # Pressure Field t=t^n+1
            T0=Function(Qt)       # Temperature Field t=t^n
            T1=Function(Qt)       # Temperature Field t=t^n+1
            tau0_vec=Function(Z)     # Stress Field (Vector) t=t^n
            tau12_vec=Function(Z)    # Stress Field (Vector) t=t^n+1/2
            tau1_vec=Function(Z)     # Stress Field (Vector) t=t^n+1
            w0= Function(W)
            w12= Function(W)
            ws= Function(W)
            w1= Function(W)
            (u0, D0_vec)=w0.split()
            (u12, D12_vec)=w0.split()
            (us, Ds_vec)=w0.split()
            (u1, D1_vec)=w0.split()
            break


        # Plot solution
        #if t>0.1:
            #plot(c1, title="SUPG Parameter", rescale=False)
            #plot(tau1[0,0], title="Normal Stress", rescale=True)
            #plot(p1, title="Pressure", rescale=True)
            #plot(u1, title="Velocity", rescale=True, mode = "auto")
            #plot(T1, title="Temperature", rescale=True)
                

        # Move to next time step
        w0.assign(w1)
        T0.assign(T1)
        p0.assign(p1)
        tau0_vec.assign(tau1_vec)
        t += dt


    if jj==1: 

        # Minimum of stream function (Eye of Rotation)
        u1_V = project(u1, V)
        psi = stream_function(u1_V)
        psi_min = min(psi.vector().get_local())
        min_loc = min_location(psi)
        with open("Mesh-Stream-Function.txt", "a") as text_file:
             text_file.write("Re="+str(Re*conv)+", We="+str(We)+", Ma="+str(Ma)+", Mesh="+str(mm)+", t="+str(t)+"----- psi_min="+str(psi_min)+"---"+str(min_loc)+'\n')

        # Data on Kinetic/Elastic Energies
        with open("Mesh-ConformEnergy.txt", "a") as text_file:
             text_file.write("Re="+str(Rey*conv)+", We="+str(We)+", Ma="+str(Ma)+", Mesh="+str(mm)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+'\n')


        # Plot Stabilisation coeff
        mplot(kapp)
        plt.colorbar()
        plt.savefig("Mesh-Stabilisation-Re="+str(Rey)+"We="+str(We)+"beta="+str(betav)+"t="+str(t)+".png")
        plt.clf()
        f_corr = File("Paraview_Results/Stabilisation_Re="+str(Re*conv)+"We="+str(We)+"beta="+str(betav)+"/kappa "+str(t)+".pvd")
        f_corr << kapp    
        #quit()

        # Plot Cross Section Flow Values 
        """sig1_vec = project(((1.0-betav)/We)*(tau1_vec - I_vec), Zc) #Extra Stress
        u_x = project(u1[0],Q)      # Project U_x onto scalar function space
        u_y = project(u1[1],Q)      # Project U_y onto scalar function space
        sig_xx = project(sig1_vec[0],Q)
        sig_xy = project(sig1_vec[1],Q)
        sig_yy = project(sig1_vec[2],Q)
        for i in range(mm):
            x_axis.append(0.5*(1.0-cos(i*pi/mm)))
            y_axis.append(0.5*(1.0-cos(i*pi/mm)))
            u_xg.append(u_x([0.5,0.5*(1.0-cos(i*pi/mm))]))   
            u_yg.append(u_y([0.5*(1.0-cos(i*pi/mm)),0.75])) 
            sig_xxg.append(sig_xx([0.5*(1.0-cos(i*pi/mm)), 1.0])) 
            sig_xyg.append(sig_xy([0.5*(1.0-cos(i*pi/mm)), 1.0])) 
            sig_yyg.append(sig_xx([0.5*(1.0-cos(i*pi/mm)), 1.0]))  
        if j==loopend:
            # First Normal Stress
            x_axis1 = list(chunks(x_axis, mm))
            y_axis1 = list(chunks(y_axis, mm))
            u_x1 = list(chunks(u_xg, mm))
            u_y1 = list(chunks(u_yg, mm))
            sig_xx1 = list(chunks(sig_xxg, mm))
            sig_yy1 = list(chunks(sig_yyg, mm))
            plt.figure(0)
            plt.plot(x_axis1[0], u_y1[0], 'r-', label=r'$We=0.1$')
            plt.plot(x_axis1[1], u_y1[1], 'b-', label=r'$We=0.25$')
            plt.plot(x_axis1[2], u_y1[2], 'c-', label=r'$We=0.5$')
            plt.plot(x_axis1[3], u_y1[3], 'm-', label=r'$We=0.75$')
            plt.plot(x_axis1[4], u_y1[4], 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('$u_y(x,0.75)$')
            plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/u_yRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            plt.figure(1)
            plt.plot(u_x1[0], y_axis1[0], 'r-', label=r'$We=0.1$')
            plt.plot(u_x1[1], y_axis1[1], 'b-', label=r'$We=0.25$')
            plt.plot(u_x1[2], y_axis1[2], 'c-', label=r'$We=0.5$')
            plt.plot(u_x1[3], y_axis1[3], 'm-', label=r'$We=0.75$')
            plt.plot(u_x1[4], y_axis1[4], 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('$u_x(0.5,y)$')
            plt.ylabel('y')
            plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/u_xRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            plt.figure(2)
            plt.plot(x_axis1[0], sig_xx1[0], 'r-', label=r'$We=0.1$')
            plt.plot(x_axis1[1], sig_xx1[1], 'b-', label=r'$We=0.25$')
            plt.plot(x_axis1[2], sig_xx1[2], 'c-', label=r'$We=0.5$')
            plt.plot(x_axis1[3], sig_xx1[3], 'm-', label=r'$We=0.75$')
            plt.plot(x_axis1[4], sig_xx1[4], 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('$\tau_{xx}(x,1.0)$')
            plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/sig_xxRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            plt.figure(3)
            plt.plot(x_axis1[0], sig_yy1[0], 'r-', label=r'$We=0.1$')
            plt.plot(x_axis1[1], sig_yy1[1], 'b-', label=r'$We=0.25$')
            plt.plot(x_axis1[2], sig_yy1[2], 'c-', label=r'$We=0.5$')
            plt.plot(x_axis1[3], sig_yy1[3], 'm-', label=r'$We=0.75$')
            plt.plot(x_axis1[4], sig_yy1[4], 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('$\tau_{yy}(x,1.0)$')
            plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/sig_yyRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            plt.close()"""


            #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re FIXED 
        """if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==loopend or j==3 or j==1:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'$We=0.1$')
            plt.plot(x2, ek2, 'b-', label=r'$We=0.25$')
            plt.plot(x3, ek3, 'c-', label=r'$We=0.5$')
            plt.plot(x4, ek4, 'm-', label=r'$We=0.75$')
            plt.plot(x5, ek5, 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_{kinetic}$')
            plt.savefig("Compressible Viscoelastic Flow Results/Energy/KineticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"mesh="+str(mm)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            plt.close()
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'$We=0.1$')
            plt.plot(x2, ee2, 'b-', label=r'$We=0.25$')
            plt.plot(x3, ee3, 'c-', label=r'$We=0.5$')
            plt.plot(x4, ee4, 'm-', label=r'$We=0.75$')
            plt.plot(x5, ee5, 'g-', label=r'$We=1.0$')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_{elastic}$')
            plt.savefig("Compressible Viscoelastic Flow Results/Energy/ElasticEnergyRe="+str(Re*conv)+"Tf="+str(Tf)+"b="+str(betav)+"mesh="+str(mm)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()"""

        """if j==loopend:
            # First Normal Stress
            x_axis1 = list(chunks(x_axis, mm))
            y_axis1 = list(chunks(y_axis, mm))
            u_x1 = list(chunks(u_xg, mm))
            u_y1 = list(chunks(u_yg, mm))
            sig_xx1 = list(chunks(sig_xxg, mm))
            sig_xy1 = list(chunks(sig_xyg, mm))
            sig_yy1 = list(chunks(sig_yyg, mm))
            plt.figure(0)
            plt.plot(x_axis1[0], u_y1[0], 'r-', label=r'$Ma=0.001$')
            plt.plot(x_axis1[1], u_y1[1], 'b-', label=r'$Ma=0.01$')
            plt.plot(x_axis1[2], u_y1[2], 'c-', label=r'$Ma=0.1$')
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('$u_y(x,0.75)$')
            plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/u_yRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            plt.figure(1)
            plt.plot(u_x1[0], y_axis1[0], 'r-', label=r'$Ma=0.001$')
            plt.plot(u_x1[1], y_axis1[1], 'b-', label=r'$Ma=0.01$')
            plt.plot(u_x1[2], y_axis1[2], 'c-', label=r'$Ma=0.1$')
            plt.legend(loc='best')
            plt.xlabel('$u_x(0.5,y)$')
            plt.ylabel('y')
            plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/u_xRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            plt.figure(2)
            plt.plot(x_axis1[0], sig_xx1[0], 'r-', label=r'$Ma=0.001$')
            plt.plot(x_axis1[1], sig_xx1[1], 'b-', label=r'$Ma=0.01$')
            plt.plot(x_axis1[2], sig_xx1[2], 'c-', label=r'$Ma=0.1$')
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('$\tau_{xx}(x,1.0)$')
            plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/tau_xxRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            plt.figure(3)
            plt.plot(x_axis1[0], sig_yy1[0], 'r-', label=r'$Ma=0.001$')
            plt.plot(x_axis1[1], sig_yy1[1], 'b-', label=r'$Ma=0.01$')
            plt.plot(x_axis1[2], sig_yy1[2], 'c-', label=r'$Ma=0.1$')
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('$\tau_{yy}(x,1.0)$')
            plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/tau_yyRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            plt.figure(4)
            plt.plot(x_axis1[0], sig_xy1[0], 'r-', label=r'$Ma=0.001$')
            plt.plot(x_axis1[1], sig_xy1[1], 'b-', label=r'$Ma=0.01$')
            plt.plot(x_axis1[2], sig_xy1[2], 'c-', label=r'$Ma=0.1$')
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('$\tau_{xy}(x,1.0)$')
            plt.savefig("Compressible Viscoelastic Flow Results/Cross Section/tau_xyRe="+str(Re*conv)+"x="+str(0.5)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+".png")
            plt.clf()
            plt.close()"""
       

         # Plot Mesh Convergence Data 
        if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E10 and j==loopend or j==2 or j==1:
            # Kinetic Energy
            plt.figure(0)
            plt.plot(x1, ek1, 'r-', label=r'M1')
            plt.plot(x2, ek2, 'b--', label=r'M2')
            plt.plot(x3, ek3, 'c-', label=r'M3')
            plt.plot(x4, ek4, 'm:', label=r'M4')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_k$')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/Mesh_KineticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
            # Elastic Energy
            plt.figure(1)
            plt.plot(x1, ee1, 'r-', label=r'M1')
            plt.plot(x2, ee2, 'b--', label=r'M2')
            plt.plot(x3, ee3, 'c-', label=r'M3')
            plt.plot(x4, ee4, 'm:', label=r'M4')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$E_e$')
            plt.savefig("Incompressible Viscoelastic Flow Results/Energy/Mesh_ElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
            plt.clf()

        # Save Data


        fv = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"/velocity "+str(t)+".pvd")
        fv_x = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"/u_x "+str(t)+".pvd")
        fv_y = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"/u_y "+str(t)+".pvd")
        fmom = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"/mom "+str(t)+".pvd")
        ftau_xx = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"/tua_xx "+str(t)+".pvd")
        ftau_xy = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"/tau_xy "+str(t)+".pvd")
        ftau_yy = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"/tau_yy "+str(t)+".pvd")
        f_N1 = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"/N1"+str(t)+".pvd")


        if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E10 and abs(E_k) < 20 and j==loopend or j==1 or j==2 or j==3:

            # Plot Stress/Normal Stress Difference
            tau_xx=project(tau1[0,0],Q)
            ftau_xx << tau_xx
            mplot(tau_xx)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/Meshtau_xxRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf() 
            tau_xy=project(tau1[1,0],Q)
            ftau_xy << tau_xy
            mplot(tau_xy)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/Meshtau_xyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf() 
            tau_yy=project(tau1[1,1],Q)
            ftau_yy << tau_yy
            mplot(tau_yy)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/Meshtau_yyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf() 


            fv << u1
     
           # Plot Velocity Components
            ux=project(u1[0],Q)
            mplot(ux)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/Meshu_xRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()
            uy=project(u1[1],Q)
            mplot(uy)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/Meshu_yRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()
            mplot(psi)
            plt.colorbar()
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/MeshpsiRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()

        if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E10 and j==loopend or j==1 or j==3:
            #Plot Contours USING MATPLOTLIB
            # Scalar Function code


            x = Expression('x[0]', degree=2)     #GET X-COORDINATES LIST
            y = Expression('x[1]', degree=2)     #GET Y-COORDINATES LIST
            pvals = p1.vector().get_local()          # GET SOLUTION p= p(x,y) list
            psiq = project(psi, Q)
            psivals = psiq.vector().get_local() 
            tauxx = project(tau1_vec[0], Q)
            tauxxvals = tauxx.vector().get_local()
            xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            xvalsq = interpolate(x, Q)#xyvals[:,0]
            yvalsq= interpolate(y, Q)#xyvals[:,1]
            xvalsw = interpolate(x, Qt)#xyvals[:,0]
            yvalsw= interpolate(y, Qt)#xyvals[:,1]

            xvals = xvalsq.vector().get_local()
            yvals = yvalsq.vector().get_local()


            xx = np.linspace(0,1)
            yy = np.linspace(0,1)
            XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
            pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') 
            psps = mlab.griddata(xvals, yvals, psivals, xx, yy, interp='nn')  


            plt.contour(XX, YY, pp, 25)
            plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/MeshPressureContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()



            """plt.contour(XX, YY, psps, 15)
            plt.title('Streamline Contours')   # PRESSURE CONTOUR PLOT
            plt.colorbar() 
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/MeshStreamlineContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
            plt.clf()"""


            #Plot Velocity Streamlines USING MATPLOTLIB
            u1_q = project(u1[0],Q)
            uvals = u1_q.vector().get_local()
            v1_q = project(u1[1],Q)
            vvals = v1_q.vector().get_local()

                # Interpoltate velocity field data onto matlab grid
            uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
            vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 


                #Determine Speed 
            speed = np.sqrt(uu*uu+ vv*vv)

            plot3 = plt.figure()
            plt.streamplot(XX, YY, uu, vv,  
                           density=2,              
                           color=speed,  
                           cmap=cm.gnuplot,                         # colour map
                           linewidth=0.8)                           # line thickness
                                                                    # arrow size
            plt.colorbar()                                          # add colour bar on the right
            plt.title('Lid Driven Cavity Problem')
            plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/MeshVelocityContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")   
            plt.clf()                                            # display the plot


        plt.close()
        err_count = 0


    if jj == 0: 
        # Calculate Stress Residual 
        F1R = Fdef(u1, tau1)  
        F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
        restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - I_vec #- diss_vec 
        res_test = inner(restau0,restau0)                            

        kapp = project(res_test, Qt) # Error Function
        norm_kapp = normalize_solution(kapp) # normalised error function

        ratio = 0.2/(1*err_count + 1.0) # Proportion of cells that we want to refine
        tau_average = project((tau1_vec[0]+tau1_vec[1]+tau1_vec[2])/3.0 , Qt)
        error_rat = project(kapp/(tau_average + 0.000001) , Qt)
        error_rat = absolute(error_rat)

        jj=1 

        if error_rat.vector().get_local().max() > 0.01 and err_count < 1:
           err_count+=1
           mesh = adaptive_refinement(mesh, norm_kapp, ratio)
           #mesh = refine_boundaries(mesh, 1)
           #mesh = refine_narrow(mesh, 1)
           mplot(error_rat)
           plt.colorbar()
           plt.savefig("adaptive-error-function.eps")
           plt.clf()
           mplot(mesh)
           plt.savefig("adaptive-mesh.eps")
           plt.clf()
           jj=0
           conv_fail = 0

        # Reset Parameters   
        dt = 20*mesh.hmin()**2
        Tf = T_f
        th = 0.5
        x1=list()
        x2=list()
        x3=list()
        x4=list()
        x5=list()
        x6=list()
        y=list()
        z=list()
        zz=list()
        zzz=list()
        zl=list()
        ek1=list()
        ek2=list()
        ek3=list()
        ek4=list()
        ee1=list()
        ee2=list()
        ee3=list()
        ee4=list()
        ek5=list()
        ee5=list()
        ek6=list()
        ee6=list()
        x_axis=list()
        y_axis=list()
        u_xg = list()
        u_yg = list()
        sig_xxg = list()
        sig_xyg = list()
        sig_yyg = list()







