import firedrake as fd
import firedrake_adjoint as fda

import torch_adjoint as fdt
import numpy as np

import pytest

def uniform_mesh(N=10):
    mesh = fd.UnitSquareMesh(N, N)

    V = fd.VectorFunctionSpace(mesh, 'CG', 1)
    Q = fd.FunctionSpace(mesh, 'CG', 1)
    W = V*Q
    return mesh, W

@pytest.mark.parallel(nprocs=2)
def test_convert_scalar_function():
    mesh, W = uniform_mesh()
    q = fd.interpolate(mesh.coordinates[0], W[1])
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    assert np.allclose(q.dat.data[:], p.dat.data[:])

@pytest.mark.parallel(nprocs=2)
def test_convert_vector_function():
    mesh, W = uniform_mesh()
    q = mesh.coordinates
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    assert np.allclose(q.dat.data[:], p.dat.data[:])

@pytest.mark.parallel(nprocs=2)
def test_convert_mixed_function():
    mesh, W = uniform_mesh()
    q = fd.Function(W)
    q.sub(0).interpolate(mesh.coordinates)
    q.sub(1).interpolate(mesh.coordinates[0])
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    for i in range(W.num_sub_spaces()):
        assert np.allclose(q.dat.data[i][:], p.dat.data[i][:])

@pytest.mark.parallel(nprocs=2)
def test_convert_vector_constant():
    q = fd.Constant((1.0, 2.0, 3.0))
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    assert np.allclose(q.dat.data[:], p.dat.data[:])

@pytest.mark.parallel(nprocs=2)
def test_convert_float():
    q = fda.AdjFloat(0.0)
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    assert np.allclose(q, p)

@pytest.mark.parallel(nprocs=2)
def test_convert_array():
    q = np.array([1., 2., 3.])
    p = fdt.util.to_firedrake(fdt.util.to_numpy(q), q)
    assert isinstance(p, type(q))
    assert np.allclose(q, p)

@pytest.mark.parallel(nprocs=2)
def test_gather_scalar():
    mesh, W = uniform_mesh()
    x = fd.SpatialCoordinate(mesh)
    u = fd.interpolate(x[0], W[1])

    u0 = fdt.util.gather(u, to_rank=0)
    u0 *= 2.0  # Scale the data to check proper transformation

    uN = fdt.util.scatter(u0, u, from_rank=0)
    assert np.allclose(u.dat.data_ro_with_halos[:], 0.5*uN.dat.data_ro_with_halos[:])

@pytest.mark.parallel(nprocs=2)
def test_gather_vector():
    mesh, W = uniform_mesh()
    x = fd.SpatialCoordinate(mesh)
    u = fd.interpolate(x, W[0])

    u0 = fdt.util.gather(u, to_rank=0)
    u0 *= 2.0  # Scale the data to check proper transformation

    uN = fdt.util.scatter(u0, u, from_rank=0)
    assert np.allclose(u.dat.data_ro_with_halos[:], 0.5*uN.dat.data_ro_with_halos[:])

@pytest.mark.parallel(nprocs=2)
def test_gather_mixed():
    mesh, W = uniform_mesh(N=2)
    x = fd.SpatialCoordinate(mesh)
    q = fd.Function(W)
    q.sub(0).interpolate(x)
    q.sub(1).interpolate(x[0])

    q0 = fdt.util.gather(q, to_rank=0)
    q0 *= 2.0  # Scale the data to check proper transformation

    qN = fdt.util.scatter(q0, q, from_rank=0)
    for i in range(W.num_sub_spaces()):
        assert np.allclose(q.dat.data_ro_with_halos[i][:], 0.5*qN.dat.data_ro_with_halos[i][:])

if __name__=="__main__":
    test_gather_mixed()