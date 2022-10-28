import numpy as np
import torch
# Import fenics and override necessary data structures with fenics_adjoint
import firedrake as fd
import firedrake_adjoint as fda
import pyadjoint

from ufl import inner, grad, dx, div, sin, pi

from torch_firedrake import FiredrakeModule

# Declare the model corresponding to solving the Poisson equation
# with variable source term and boundary value
class Poisson(FiredrakeModule):
    # Construct variables which can be reused in the constructor
    def __init__(self):
        super().__init__()

        # Create function space
        mesh = fd.UnitIntervalMesh(20)
        self.V = fd.FunctionSpace(mesh, 'CG', 1)

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


class Stokes(FiredrakeModule):
    def __init__(self):
        super(Stokes, self).__init__()
        mesh = fd.UnitSquareMesh(3, 3)

        self.V = fd.VectorFunctionSpace(mesh, 'CG', 2)
        self.Q = fd.FunctionSpace(mesh, 'CG', 1)
        self.W = self.V * self.Q

        self.V, self.Q = self.W.split()

        noslip_bc = fd.DirichletBC(self.V, fd.Constant((0, 0)), (3, 4))

        x, y = fd.SpatialCoordinate(mesh)
        inflow_bc = fd.DirichletBC(self.V, (-sin(y*pi), 0.0), 1)
        outlet_bc = fd.DirichletBC(self.Q, fd.Constant(0), 2)

        self.bcs = [noslip_bc, inflow_bc, outlet_bc]

    def input_templates(self):
        return fd.Constant(0, domain=self.W.ufl_domain()), fd.Constant(0, domain=self.W.ufl_domain())


    def solve(self, fx, fy):
        f = fx*fd.Constant((1, 0)) + fy*fd.Constant((0, 1))
        u, p = fd.TrialFunctions(self.W)
        v, q = fd.TestFunctions(self.W)
        a = (inner(grad(u), grad(v)) - div(v) * p + q * div(u)) * dx
        L = inner(f, v) * dx

        w = fd.Function(self.W)
        fd.solve(a == L, w, self.bcs)

        u, p = w.split()
        return fd.assemble(inner(u, u)*dx)


def torch_grads(model, inp):
    J = model(*inp)
    J.backward()
    return tuple([x.grad for x in inp])

def pyadjoint_grads(model, inp):
    # Create pyadjoint variables to track instead of torch tensors
    x = model.input_templates()
    for i in range(len(x)):
        x[i].assign(inp[i])

    J = model.solve(*x)
    controls = [fda.Control(ctrl) for ctrl in x]
    grads = pyadjoint.compute_gradient(J, controls)
    return tuple([dJdx.values() for dJdx in grads])


def test_poisson_grads():
    # Construct the Firedrake model
    model = Poisson()

    f0 = np.random.randn()
    g0 = np.random.randn()

    # Compute gradients with PyTorch
    f = torch.tensor([[f0]], requires_grad=True, dtype=torch.float64)
    g = torch.tensor([[g0]], requires_grad=True, dtype=torch.float64)
    dJ_torch = torch_grads(model, (f, g))

    # Compute gradients with Pyadjoint
    dJ_fda = pyadjoint_grads(model, (f0, g0))

    for tg, fg in zip(dJ_torch, dJ_fda):
        assert np.allclose(tg.detach().numpy(), fg)


def test_stokes_grads():
    # Construct the Firedrake model
    model = Stokes()

    # firedrake_adjoint isn't able to handle vector-valued constants,
    #   so we have to break it up
    fx0 = 1.0
    fy0 = 1.0
    fx = torch.tensor([[fx0]], requires_grad=True, dtype=torch.float64)
    fy = torch.tensor([[fy0]], requires_grad=True, dtype=torch.float64)

    # Compute gradients with PyTorch
    dJ_torch = torch_grads(model, (fx, fy))

    # Compute gradients with Pyadjoint
    dJ_fda = pyadjoint_grads(model, (fx0, fy0))

    for tg, fg in zip(dJ_torch, dJ_fda):
        assert np.allclose(tg.detach().numpy(), fg)


if __name__=="__main__":
    test_stokes_grads()