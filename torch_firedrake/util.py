import firedrake as fd
import firedrake_adjoint as fda
import numpy as np


def to_np(fd_var):
    """Convert Firedrake variable to numpy array"""
    if isinstance(fd_var, (fd.Constant, fda.Constant)):
        return fd_var.values()

    if isinstance(fd_var, (fd.Function, fda.Constant)):
        np_array = fd_var.vector().get_local()
        n_sub = fd_var.function_space().num_sub_spaces()
        # Reshape if function is multi-component
        if n_sub != 0:
            np_array = np.reshape(np_array, (len(np_array) // n_sub, n_sub))
        return np_array

    if isinstance(fd_var, fd.GenericVector):
        return fd_var.get_local()

    if isinstance(fd_var, fda.AdjFloat):
        return np.array(float(fd_var), dtype=np.float_)

    raise ValueError('Cannot convert ' + str(type(fd_var)))