import pytest

import firedrake as fd
import firedrake_adjoint as fda

import torch
import numpy as np

from torch_adjoint import FiredrakeModule
from ufl import dx, inner, grad, div, sin, pi
from pyadjoint import create_overloaded_object

class Squares(FiredrakeModule):
    def __init__(self):
        super(Squares, self).__init__()
        mesh = fd.IntervalMesh(4, 0, 1)
        self.V = fd.FunctionSpace(mesh, 'DG', 0)

    def solve(self, f1, f2):
        u = fd.TrialFunction(self.V)
        v = fd.TestFunction(self.V)

        a = u * v * dx
        L = f1**2 * f2**2 * v * dx

        u_ = fd.Function(self.V)
        fd.solve(a == L, u_)

        return u_

    def input_templates(self):
        return fd.Function(self.V), fd.Function(self.V)

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

class DoublePoisson(FiredrakeModule):
    def __init__(self):
        super(DoublePoisson, self).__init__()
        mesh = fd.UnitIntervalMesh(10)
        self.V = fd.FunctionSpace(mesh, 'P', 1)
        self.bc = fd.DirichletBC(self.V, fd.Constant(0), 'on_boundary')

    def solve(self, f1, f2):
        u = fd.TrialFunction(self.V)
        v = fd.TestFunction(self.V)

        a = inner(grad(u), grad(v)) * dx
        L1 = f1 * v * dx
        L2 = f2 * v * dx

        u1 = fd.Function(self.V)
        fd.solve(a == L1, u1, self.bc)

        u2 = fd.Function(self.V)
        fd.solve(a == L2, u2, self.bc)

        return u1, u2, f1, f2

    def input_templates(self):
        return fd.Constant(0), fd.Constant(0)


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
        return create_overloaded_object(np.array([0, 0]))

    def solve(self, f):
        f = fd.Constant(f[0])*fd.Constant((1, 0)) + fd.Constant(f[1])*fd.Constant((0, 1))
        u, p = fd.TrialFunctions(self.W)
        v, q = fd.TestFunctions(self.W)
        a = (inner(grad(u), grad(v)) - div(v) * p + q * div(u)) * dx
        L = inner(f, v) * dx

        w = fd.Function(self.W)
        fd.solve(a == L, w, self.bcs)

        u, p = w.split()
        return fd.assemble(inner(u, u)*dx)

def test_squares():
    f1 = torch.autograd.Variable(torch.tensor([[1, 2, 3, 4],
                                               [2, 3, 5, 6]]).double(), requires_grad=True)
    f2 = torch.autograd.Variable(torch.tensor([[2, 3, 5, 6],
                                               [1, 2, 2, 1]]).double(), requires_grad=True)

    rank = fd.COMM_WORLD.Get_rank()
    size = fd.COMM_WORLD.Get_size()
    f1 = f1[:,rank::size]
    f2 = f2[:,rank::size]

    squares = Squares()

    assert np.all((squares(f1, f2) == f1**2 * f2**2).detach().numpy())
    assert torch.autograd.gradcheck(squares, (f1, f2))


@pytest.mark.skipif(fd.COMM_WORLD.size > 1, reason='Running with MPI')
def test_poisson():
    f = torch.tensor([[1.0]], requires_grad=True).double()
    g = torch.tensor([[2.0]], requires_grad=True).double()
    poisson = Poisson()
    assert torch.autograd.gradcheck(poisson, (f, g))


@pytest.mark.skipif(fd.COMM_WORLD.size > 1, reason='Running with MPI')
def test_doublepoisson():
    f1 = torch.tensor([[1.0]], requires_grad=True).double()
    f2 = torch.tensor([[2.0]], requires_grad=True).double()
    double_poisson = DoublePoisson()
    assert torch.autograd.gradcheck(double_poisson, (f1, f2))

@pytest.mark.skipif(fd.COMM_WORLD.size > 1, reason='Running with MPI')
def test_stokes():
    f = torch.tensor([[1.0, 1.0]], requires_grad=True, dtype=torch.float64)
    stokes = Stokes()
    assert torch.autograd.gradcheck(stokes, (f,))

@pytest.mark.skipif(fd.COMM_WORLD.size > 1, reason='Running with MPI')
def test_stokes_pyadjoint():
    stokes = Stokes()
    f = stokes.input_templates()
    f[:] = np.array([1, 1])
    J = stokes.solve(f)
    dJdm = fda.compute_gradient(J, fda.Control(f))
    print(dJdm)
    

@pytest.mark.skipif(fd.COMM_WORLD.size > 1, reason='Running with MPI')
def test_input_type():
    f = np.array([[1.0]])
    g = np.array([[0.0]])
    poisson = Poisson()
    poisson(f, g)
    with pytest.raises(TypeError):
        f = np.array([[1.0]], dtype=np.float32)
        g = np.array([[0.0]], dtype=np.float32)
        poisson(f, g)

    f = torch.tensor([[1.0]]).double()
    g = torch.tensor([[0.0]]).double()
    poisson(f, g)
    with pytest.raises(TypeError):
        f = torch.tensor([[1.0]]).float()
        g = torch.tensor([[0.0]]).float()
        poisson(f, g)


if __name__=="__main__":
    test_stokes_pyadjoint()