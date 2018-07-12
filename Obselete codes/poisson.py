""" Fenics Tutorial demo program: Poisson equation with Dirichlet conditions """

from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.0)
a = inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx

print(a)

A = assemble(a)
b = assemble(L)

print(A)
print(b.sum())


# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)
interactive()
