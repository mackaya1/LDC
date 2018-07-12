from __future__ import print_function
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
from ufl import *
import numpy as np
# Create Geometry

x0 = 0.0
y0 = 0.0
x1 = 1.0
y1 = 1.0
nx = 10
ny = 10

mesh = UnitSquareMesh(nx, ny)

# Save to file and plot
File("rectangle.pvd") << mesh
plot(mesh, interactive=True)

# Time Step Conditions

T_stop = 2.0 

P = FunctionSpace(mesh, "Lagrange", 2)
T = TensorFunctionSpace(mesh, "Lagrange", 2)
U = VectorFunctionSpace(mesh, "Lagrange", 2)
element = (U, T, P)
(v, vs, vp) = TestFunction(element)
(u, s,  p)  = TrialFunction(element)
(fu, fs, fp) = Function(element)
(uc, sc, pc) = Function(element)
(du, ds, dp) = Function(element)
lam = Constant(0.1) #characteristic relaxation time
eta = Constant(0.01) #viscosity
etaE = Constant(0.01)
dt = Constant(0.01)
theta = Constant(10)

# Define boundary conditions
noslip  = DirichletBC(element.sub(0), (0.0, 0.0), "on_boundary") #Velocity boundary condition 1
bcu = [noslip]



# Strain rate

def D(q):
    return 0.5*(grad(q) + transp(grad(q)))
sthatlin = -mult(transp(grad(uc)),ms) - mult(transp(grad(u)),msc) \
         - mult(ms,grad(uc)) - mult(msc,grad(u))


#Upper convected model
sthat = s3 - mult(transp(grad(uc)),msc) - mult(msc,grad(uc))


# Bilinear forms
a1 = lam*dot(ms,mvs)*dx + theta*dt*lam*dot(s1,mvs)*dx  + theta*dt*lam*dot(s2,mvs)*dx \
   + theta*dt*lam*dot(sthatlin,mvs)*dx + theta*dt*dot(ms,mvs)*dx - theta*dt*2*eta*dot(D(u),mvs)*dx
a2 = -p*div(v)*dx + 2*etaE*dot(D(u),grad(v))*dx + dot(ms,grad(v))*dx
a3 =  div(u)*vp*dx
a =   a1 + a2 + a3


# Linear forms
L1 = theta*dt*lam*dot(sthat,mvs)*dx + theta*dt*dot(msc,mvs)*dx \
   - theta*dt*2*eta*dot(D(uc),mvs)*dx +theta*dt*dot(mds,mvs)*dx
L2 = dot(fu, v)*dx + pc*div(v)*dx - 2*etaE*dot(D(uc),grad(v))*dx + dot(msc,grad(v))*dx
L3 = div(uc)*vp*dx
L  = L2 - L1 - L3

# Solve Equations 

A = assemble(a)

solve(A, z.vector(), b, "bicgstab", "default")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:

    # Update pressure boundary condition
    p_in.t = t

    # Compute tentative velocity step
    begin("Computing tentative velocity")
    b = assemble(L)
    [bc.apply(A, b) for bc in bcu]
    solve(A, u.vector(), b1, "bicgstab", "default")
    end()
 

    # Plot solution
    plot(p1, title="Pressure", rescale=True)
    plot(u1, title="Velocity", rescale=True)

    # Save to file
    ufile << u1
    pfile << p1

    # Move to next time step
    u0.assign(u1)
    t += dt
    print("t =", t)

# Hold plot
interactive()


