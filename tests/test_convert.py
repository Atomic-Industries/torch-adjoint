import firedrake as fd
import firedrake_adjoint as fda

import torch_firedrake as fdt
import numpy as np

mesh = fd.UnitSquareMesh(10, 10)

V = fd.VectorFunctionSpace(mesh, 'CG', 1)
Q = fd.FunctionSpace(mesh, 'CG', 1)
W = V*Q

def test_convert_scalar():
    q = fd.interpolate(mesh.coordinates[0], Q)
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert type(q) == type(p)
    assert np.allclose(q.dat.data[:], p.dat.data[:])

def test_convert_vector():
    q = mesh.coordinates
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert type(q) == type(p)
    assert np.allclose(q.dat.data[:], p.dat.data[:])

def test_convert_mixed():
    q = fd.Function(W)
    q.sub(0).interpolate(mesh.coordinates)
    q.sub(1).interpolate(mesh.coordinates[0])
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert type(q) == type(p)
    for i in range(W.num_sub_spaces()):
        assert np.allclose(q.dat.data[i][:], p.dat.data[i][:])

def test_convert():
    q = fd.Constant((1.0, 2.0, 3.0))
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert type(q) == type(p)
    assert np.allclose(q.dat.data[:], p.dat.data[:])

def test_convert_float():
    q = fda.AdjFloat(0.0)
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert type(q) == type(p)
    assert np.allclose(q, p)


