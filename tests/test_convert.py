import firedrake as fd
import firedrake_adjoint as fda

import torch_adjoint as fdt
import numpy as np

from pyadjoint import create_overloaded_object

mesh = fd.UnitSquareMesh(10, 10)

V = fd.VectorFunctionSpace(mesh, 'CG', 1)
Q = fd.FunctionSpace(mesh, 'CG', 1)
W = V*Q

def test_convert_scalar_function():
    q = fd.interpolate(mesh.coordinates[0], Q)
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    assert np.allclose(q.dat.data[:], p.dat.data[:])

def test_convert_vector_function():
    q = mesh.coordinates
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    assert np.allclose(q.dat.data[:], p.dat.data[:])

def test_convert_mixed_function():
    q = fd.Function(W)
    q.sub(0).interpolate(mesh.coordinates)
    q.sub(1).interpolate(mesh.coordinates[0])
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    for i in range(W.num_sub_spaces()):
        assert np.allclose(q.dat.data[i][:], p.dat.data[i][:])

def test_convert_vector_constant():
    q = fd.Constant((1.0, 2.0, 3.0))
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    assert np.allclose(q.dat.data[:], p.dat.data[:])

def test_convert_float():
    q = fda.AdjFloat(0.0)
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    assert np.allclose(q, p)

def test_convert_array():
    q = np.array([1., 2., 3.])
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    assert np.allclose(q, p)

