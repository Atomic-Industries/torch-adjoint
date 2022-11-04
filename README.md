# Torch-adjoint

This is a fork of Torch-FEniCS modified for [Firedrake](https://www.firedrakeproject.org/) and MPI-parallelization.
The `torch-adjoint` package enables models defined in Firedrake (https://fenicsproject.org) to be used as modules in [PyTorch](https://pytorch.org/).

## Install

[Install Firedrake](https://https://www.firedrakeproject.org/download.html) and run

```bash
pip install git+https://github.com/Atomic-Industries/torch-adjoint.git@master
```

Alternatively, a simple Dockerfile is also provided that builds off the official Firedrake images.
This can be built and run with `make build && make bash`.  Be sure to activate the Firedrake virtual environment
with `source activate /home/firedrake/firedrake/bin/activate`.

## Details

Firedrake objects are represented in PyTorch using their corresponding vector representation. For 
finite element functions this corresponds to their coefficient representation.
This package interfaces between the automatic differentiation frameworks in
[`dolfin-adjoint`](http://www.dolfin-adjoint.org/en/latest/) and PyTorch so that the gradient tapes
can understand each other.  See `torch_adjoint.SolveFunction.backward()` for the core logic that does this.

The package currently handles MPI-parallelization only on the Firedrake side.  That is, Firedrake distributes mesh nodes
across processors, and then `torch-adjoint` gathers the data back to rank-0, where all of the PyTorch tensors (including
model parameters) live.  This means that there must be enough memory available to the rank-0 process to hold all the
Firedrake data.  Support for distributed training with MPI and PyTorch is planned in the future.

## Example

The `torch-adjoint` package can for example be used to define a PyTorch module which solves the Poisson
equation using Firedrake.

The process of solving the Poisson equation in Firedrake can be specified as a PyTorch module by subclassing `torch_adjoint.FiredrakeModule`

```python
# Import PyTorch, Firedrake and useful math from UFL
import torch
import firedrake as fd
from ufl import inner, grad, dx

from torch_adjoint import FiredrakeModule

# Declare the Firedrake model corresponding to solving the Poisson equation
# with variable source term and boundary value
class Poisson(FiredrakeModule):
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
        # Construct linear form from input
        L = f * self.v * dx

        # Construct boundary condition from input
        bc = fd.DirichletBC(self.V, g, 'on_boundary')

        # Solve the Poisson equation
        u = fd.Function(self.V)
        fd.solve(self.a == L, u, bc)

        return u

    def input_templates(self):
        # Declare templates for the inputs to Poisson.solve.
        #   Note that for constants the the mesh has to be passed to the constructor
        #   In order for UFL to recognize the domain the constant belongs to.
        return fd.Constant(0, domain=self.V.ufl_domain()), fd.Constant(0, domain=self.V.ufl_domain())
```

The `Poisson.solve` function can now be executed by giving the module 
the appropriate vector input corresponding to the input templates declared in 
`Poisson.input_templates`. In this case the vector representation of the 
template `Constant(0)` is simply a scalar. 

```python
# Construct the PDE model
poisson = Poisson()

# Create N sets of input
N = 10
f = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)
g = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)

# Solve the Poisson equation N times
u = poisson(f, g)
```

The output of the can now be used to construct some functional. Consider summing
up the coefficients of the solutions to the Poisson equation

```python
# Construct functional 
J = u.sum()
```

The derivative of this functional with respect to `f` and `g` can now be
computed using the `torch.autograd` framework.

```python
# Execute backward pass
J.backward() 

# Extract gradients
dJdf = f.grad
dJdg = g.grad
```

## Developing
Some of the tests are set up to test MPI-parallelization

```bash
mpiexec -np 3 pytest tests
```
