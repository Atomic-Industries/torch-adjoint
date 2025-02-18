import numpy as np
import torch
import firedrake as fd
import firedrake_adjoint as fda
import pyadjoint
import pytest

from ufl import inner, grad, dx, div, sin, pi

from torch_adjoint import FiredrakeModule

class Poisson(FiredrakeModule):
    def __init__(self):
        super().__init__()
        mesh = fd.UnitIntervalMesh(20)
        self.V = fd.FunctionSpace(mesh, 'CG', 1)
        u = fd.TrialFunction(self.V)
        self.v = fd.TestFunction(self.V)
        self.a = inner(grad(u), grad(self.v)) * dx

    def solve(self, f, g):
        L = f * self.v * dx
        bc = fd.DirichletBC(self.V, g, 'on_boundary')
        u = fd.Function(self.V)
        fd.solve(self.a == L, u, bc)
        return fd.assemble(inner(u, u)*dx)

    def input_templates(self):
        return fd.Constant(0, domain=self.V.ufl_domain()), fd.Constant(0, domain=self.V.ufl_domain())

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


@pytest.mark.parallel(nprocs=2)
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


if __name__=="__main__":
    test_poisson_grads()