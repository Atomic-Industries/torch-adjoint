import firedrake as fd
import firedrake_adjoint as fda

import torch_adjoint as fdt
import numpy as np

import pytest

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

@pytest.mark.parallel(nprocs=2)
def test_gather_scalar():
    N = 10
    mesh = fd.UnitSquareMesh(N, N)
    x = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, 'CG', 1)
    u = fd.interpolate(x[0], V)

    u0 = fdt.util.gather(u, to_rank=0)
    u0.array *= 2.0  # Scale the data to check proper transformation

    uN = fdt.util.scatter(u0, u, from_rank=0)
    assert np.allclose(u.dat.data_ro_with_halos[:], 0.5*uN.dat.data_ro_with_halos[:])

@pytest.mark.parallel(nprocs=2)
def test_gather_vector():
    N = 10
    mesh = fd.UnitSquareMesh(N, N)
    x = fd.SpatialCoordinate(mesh)

    V = fd.VectorFunctionSpace(mesh, 'CG', 1)
    u = fd.interpolate(x, V)

    u0 = fdt.util.gather(u, to_rank=0)
    u0.array *= 2.0  # Scale the data to check proper transformation

    uN = fdt.util.scatter(u0, u, from_rank=0)
    assert np.allclose(u.dat.data_ro_with_halos[:], 0.5*uN.dat.data_ro_with_halos[:])
