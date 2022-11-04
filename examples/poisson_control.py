r""" Solves a optimal control problem constrained by the Poisson equation:
    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2
    subject to
    grad \cdot \grad u = f    in \Omega
    u = 0                     on \partial \Omega
"""
import torch
import firedrake as fd

from ufl import inner, grad, dx, sin, pi

from firedrake import logging
from firedrake.logging import logger
logging.set_log_level(logging.INFO)

from torch_adjoint import FiredrakeModule
from torch_adjoint.util import gather_and_reshape, to_firedrake


class MLP(torch.nn.Module):
    """
    A simple neural network to learn
    a scalar spatial function on mesh vertices.

    Relu activation on the hidden layers, sigmoid on output
    """

    def __init__(self, mesh, num_hidden_layers=2, hidden_layers_size=50, activation=torch.tanh):
        super().__init__()

        self.act = activation

        self.input_layer = torch.nn.Linear(
            mesh.geometric_dimension(), hidden_layers_size, dtype=torch.double
        )

        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_layers_size, hidden_layers_size, dtype=torch.double) for _ in range(num_hidden_layers)
        ])

        self.output = torch.nn.Linear(hidden_layers_size, 1, dtype=torch.double)

    def forward(self, x):
        """
        Input a set of position vectors to query for the material density
        at those locations.
        """
        x = self.act(self.input_layer(x))
        for linear in self.hidden_layers:
            x = self.act(linear(x))
        return self.output(x)


# Declare the model corresponding to solving the Poisson equation
# with variable source term and boundary value
class Poisson(FiredrakeModule):
    def __init__(self, N=64, alpha=1e-6):
        super().__init__()
        self.alpha = fd.Constant(alpha)

        # Create function space
        mesh = fd.UnitSquareMesh(N, N)
        self.V = fd.FunctionSpace(mesh, 'CG', 1)

        # Create trial and test functions
        u = fd.TrialFunction(self.V)
        self.v = fd.TestFunction(self.V)

        # Construct bilinear form
        self.a = inner(grad(u), grad(self.v)) * dx

        self.set_opt_target()

    def set_opt_target(self):
        x = fd.SpatialCoordinate(self.V.ufl_domain())
        w = sin(pi*x[0])*sin(pi*x[1])
        self.u_target = w / (2 * pi**2)

        # Define the expressions of the analytical solution
        self.f_analytic = w/(1 + self.alpha*4*pi**4)
        self.u_analytic = self.f_analytic/(2*pi**2)

        self.J = lambda u, f: fd.assemble(inner(u-self.u_target, u-self.u_target)*dx + 0.5*self.alpha*f**2*dx)

    def solve_poisson(self, f):
        # Construct linear form
        L = f * self.v * dx

        # Construct boundary condition
        bc = fd.DirichletBC(self.V, 0.0, 'on_boundary')

        # Solve the Poisson equation
        u = fd.Function(self.V)
        fd.solve(self.a == L, u, bc)
        return u

    def solve(self, f):
        u = self.solve_poisson(f)

        # Return the functional
        return self.J(u, f)

    def input_templates(self):
        # Declare templates for the inputs to Poisson.solve
        return fd.Function(self.V)


class ParameterVec(torch.nn.Module):
    "A dummy module for optimizing a specific vector"

    def __init__(self, vec):
        super().__init__()
        self.vec = torch.nn.Parameter(torch.empty((*vec.shape, 1), dtype=torch.float64))
        print(self.vec.shape)

    def forward(self, x):
        return self.vec

if __name__ == '__main__':
    # Construct the Firedrake model
    poisson = Poisson()
    mesh = poisson.V.ufl_domain()

    # Construct the neural net
    model = MLP(
        mesh,
        num_hidden_layers=1,
        hidden_layers_size=1024,
        activation=torch.relu
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    x = torch.tensor(
        gather_and_reshape(mesh.coordinates),
        dtype=torch.float64,
        requires_grad=True
    )

    opt = torch.optim.LBFGS(model.parameters(), lr=1)
    # opt = torch.optim.Adam(model.parameters(), lr=1e-6)

    n_steps = 1000
    tol = 1e-6
    pvd = fd.File('output/poisson_torch.pvd')
    prevJ = 1e10
    for i in range(n_steps):

        def closure():
            opt.zero_grad()
            f = model(x).t()
            J = poisson(f)
            J.backward()
            return J

        opt.step(closure)

        # Log progress
        f_opt = to_firedrake(model(x).detach().numpy(), poisson.input_templates())
        u_opt = poisson.solve_poisson(f_opt)
        control_error = fd.errornorm(poisson.f_analytic, f_opt)
        state_error = fd.errornorm(poisson.u_analytic, u_opt)
        J = poisson.J(u_opt, f_opt)
        logger.info(f"Step {i+1}/{n_steps}: J={J} \t ||u-u_a||={state_error} \t ||f-f_a||={control_error}")

        dJ = 1 - J/prevJ
        done = abs(1 - J/prevJ) < tol
        prevJ = J

        if i%1 == 0 or done:
            f_opt.rename('f')
            f_analytic = fd.interpolate(poisson.f_analytic, poisson.V)
            f_analytic.rename('f_a')
            u_opt.rename('u')
            u_analytic = fd.interpolate(poisson.u_analytic, poisson.V)
            u_analytic.rename('u_a')
            u_target = fd.interpolate(poisson.u_target, poisson.V)
            u_target.rename('u_t')
            pvd.write(f_opt, f_analytic, u_opt, u_analytic, u_target)

        if done:
            break