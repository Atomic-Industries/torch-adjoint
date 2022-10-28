import firedrake as fd
import firedrake_adjoint as fda
import numpy as np

from firedrake.functionspaceimpl import MixedFunctionSpace

def to_numpy(q):
    """Convert variable `q` to numpy array"""

    if isinstance(q, (fd.Constant, fd.Function)):
        return q.dat.data[:]

    if isinstance(q, fda.AdjFloat):
        return np.array(float(q), dtype=np.float_)

    raise ValueError('Cannot convert ' + str(type(q)))

def to_firedrake(x, var_template):
    """Convert numpy array to FEniCS variable"""

    if isinstance(var_template, fda.AdjFloat):
        return fda.AdjFloat(x)

    if isinstance(var_template, fd.Constant):
        if x.shape == (1,):
            x = x[0]
        return fd.Constant(x, domain=var_template.ufl_domain())

    if isinstance(var_template, fd.Function):
        function_space = var_template.function_space()
        u = type(var_template)(function_space)

        if isinstance(function_space.topological, MixedFunctionSpace):
            for i in range(function_space.num_sub_spaces()):
                u.dat.data[i][:] = x[i]
        else:
            u.dat.data[:] = x
            
        return u

    err_msg = 'Cannot convert numpy array to {}'.format(var_template)
    raise ValueError(err_msg)