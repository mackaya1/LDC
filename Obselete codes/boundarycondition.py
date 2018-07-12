"""Boundary Condition for spinning journal"""

# Source term
class Source(Expression):
    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) + 1.0*exp(-(dx*dx + dy*dy)/0.02)

# Sub domain for Periodic boundary condition
class SpinningBoundary(c1_boundary):

    # inner cylinder spins with constant angular velocity 
      def inside(self, x, c1_boundary):
          return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

# Create periodic boundary condition
pbc = SpinningBoundary()
