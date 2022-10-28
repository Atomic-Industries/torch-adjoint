import torch
import firedrake as fd

from ufl import inner, grad, dx
from torch_adjoint import FiredrakeModule

# Declare the model corresponding to solving the Poisson equation
# with variable source term and boundary value
class Poisson(FiredrakeModule):
    # Construct variables which can be reused in the constructor
    def __init__(self):
        super().__init__()

        # Create function space
        mesh = fd.UnitIntervalMesh(20)
        self.V = fd.FunctionSpace(mesh, 'P', 1)

        # Create trial and test functions
        u = fd.TrialFunction(self.V)
        self.v = fd.TestFunction(self.V)

        # Construct bilinear form
        self.a = inner(grad(u), grad(self.v)) * dx

    def solve(self, f, g):
        # Construct linear form
        L = f * self.v * dx

        # Construct boundary condition
        bc = fd.DirichletBC(self.V, g, 'on_boundary')

        # Solve the Poisson equation
        u = fd.Function(self.V)
        fd.solve(self.a == L, u, bc)

        # Return the functional (could also return `u`)
        return fd.assemble(inner(u, u)*dx)

    def input_templates(self):
        # Declare templates for the inputs to Poisson.solve
        return fd.Constant(0, domain=self.V.ufl_domain()), fd.Constant(0, domain=self.V.ufl_domain())


if __name__ == '__main__':
    # Construct the Firedrake model
    model = Poisson()

    # Create N sets of input (Poisson will be solved N times)
    N = 10
    f = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)
    g = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)

    # Solve Poisson N times and take the mean
    J = model(f, g).mean()

    # Compute gradients with backwards pass
    J.backward()
    dJdf = f.grad
    dJdg = g.grad

    print(dJdf)
    print(dJdg)