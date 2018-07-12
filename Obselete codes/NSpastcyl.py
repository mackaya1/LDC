"""Code solving the Navier Stokes equations for a driven flow past a cylinder"""

from __future__ import print_function
from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
import numpy as np

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;


# Construct hollow cylinder mesh
r_a=0.7
r_b=2.0

# Create geometry 1
x0=1.0
y0=1.0
c1=Circle(Point(x0,y0), r_a)

# Create Rectangle

x1=0.0
y1=0.0
x2=2.0
y2=2.0

rec = RectangleMesh((x1,y1), (x2,y2), 10, 10)

shape = rec - c1 


# Create mesh
mesh = generate_mesh(shape, 50)


# Define function spa


# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Define coefficients 
k = Constant(dt)
f = Constant((0, -5))

# Tentative velocity step
F1 = inner(u - u0, v)*dx + k*inner(grad(u0)*u0, v)*dx + \
     k*nu*inner(grad(u), grad(v))*dx - k*inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update 
a2 = k*inner(grad(p), grad(q))*dx #+ eps*inner(p,q)*dx
L2 = -div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx


# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:

    # Update pressure boundary condition
    #p_in.t = t

    # Compute tentative velocity step
    begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "bicgstab", "default")
    end()

    # Pressure correction
    begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    [bc.apply(p1.vector()) for bc in bcp]
    solve(A2, p1.vector(), b2, "bicgstab", prec)
    end()

    # Velocity correction
    begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "bicgstab", "default")
    end()

    # Plot solution
    plot(p1, title="Pressure", rescale=True)
    plot(u1, title="Velocity", rescale=True, mode = "auto")


    # Save to file
    ufile << u1
    pfile << p1

    # Move to next time step
    u0.assign(u1)
    t += dt
    print("t =", t)

# Hold plot
plot(mesh)
interactive()
