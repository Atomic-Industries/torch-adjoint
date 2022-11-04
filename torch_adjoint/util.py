import firedrake as fd
import firedrake_adjoint as fda
import numpy as np
import pyadjoint

from firedrake.petsc import PETSc
from firedrake.functionspaceimpl import MixedFunctionSpace


def gather(f, to_rank=None):
    """Gather a `Function` to all processes (or a specified rank)

    Modified from the `Vector` source code.
    """
    N = f.vector().size()
    if to_rank is not None and PETSc.COMM_WORLD.rank != to_rank:
        N = 0
    
    v = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
    is_ = PETSc.IS().createStride(N, 0, 1, comm=PETSc.COMM_SELF)

    with f.dat.vec_ro as vec:
        vscat = PETSc.Scatter().create(vec, is_, v, None)
        vscat.scatterBegin(vec, v, addv=PETSc.InsertMode.INSERT_VALUES)
        vscat.scatterEnd(vec, v, addv=PETSc.InsertMode.INSERT_VALUES)
    return v.array

def gather_and_reshape(q):
    """Combine a gather operation with a reshape to match the underlying data.

    Note this should not be called directly if q belongs to a MixedFunctionSpace
    (instead call separately for each subspace).
    """
    x = gather(q, to_rank=0)
    return np.reshape(x, [-1, *q.function_space().shape])

def scatter(x, function_template, from_rank=None):
    """Scatter a `numpy.ndarray` to a `Function` on all processes
    """
    u_out = function_template.copy(deepcopy=True)

    N = u_out.vector().size()
    if from_rank is not None and PETSc.COMM_WORLD.rank != from_rank:
        N = 0

    # Create a PETSc.Vec from the numpy array
    v = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
    v.array = x

    is_ = PETSc.IS().createStride(N, 0, 1, comm=PETSc.COMM_SELF)

    with u_out.dat.vec as vec:
        vscat = PETSc.Scatter().create(vec, is_, v, None)
        vscat.scatterBegin(v, vec, addv=PETSc.InsertMode.INSERT_VALUES, mode=vscat.Mode.REVERSE)
        vscat.scatterEnd(v, vec, addv=PETSc.InsertMode.INSERT_VALUES, mode=vscat.Mode.REVERSE)
    return u_out

def reshape_and_scatter(x, q):
    """Inverse of `gather_and_reshape`.
    
    See note on MixedFunctionSpaces
    """
    qN = scatter(x.ravel(), q, from_rank=0)  # Copy of Function q with data from x
    return qN.dat.data[:]

def to_numpy(q):
    """Convert variable `q` to numpy array"""

    if isinstance(q, fd.Constant):
        return q.dat.data[:]

    if isinstance(q, fd.Function):
        function_space = q.function_space()

        # If MixedFunctionSpace, convert each component independently
        if isinstance(function_space.topological, MixedFunctionSpace):
            x = tuple([gather_and_reshape(q.sub(i)) for i in range(function_space.num_sub_spaces())])
        else:
            x = gather_and_reshape(q)

        return  x # Return the numpy data reshaped to match the original data

    if isinstance(q, fda.AdjFloat):
        return np.array(float(q), dtype=np.float_)

    if isinstance(q, np.ndarray):
        return q

    raise ValueError('Cannot convert ' + str(type(q)))

def to_firedrake(x, var_template):
    """Convert numpy array to Firedrake variable"""

    if isinstance(var_template, fd.Constant):
        if x.shape == (1,):
            x = x[0]
        return fd.Constant(x, domain=var_template.ufl_domain())

    if isinstance(var_template, fd.Function):
        function_space = var_template.function_space()
        u = type(var_template)(function_space)

        if isinstance(function_space.topological, MixedFunctionSpace):
            for i in range(function_space.num_sub_spaces()):
                u.dat.data[i][:] = reshape_and_scatter(x[i], u.sub(i))
        else:
            u.dat.data[:] = reshape_and_scatter(x, u)
            
        return u

    if isinstance(var_template, pyadjoint.AdjFloat):
        return pyadjoint.AdjFloat(x)

    if isinstance(var_template, np.ndarray):
        return pyadjoint.create_overloaded_object(np.array(x, dtype=np.float_))

    err_msg = 'Cannot convert numpy array to {}'.format(var_template)
    raise ValueError(err_msg)