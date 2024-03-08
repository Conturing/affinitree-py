import numpy as np
import pytest

from affinitree import AffTree, AffFunc, Polytope


def test_constructor_row_mismatch():
    mat = np.zeros((10, 12))
    bias = np.zeros(12)
    with pytest.raises(Exception):
        f = AffFunc.from_mats(mat, bias)


def test_constructor_shape_mat():
    mat = np.zeros((10, 12, 2))
    bias = np.zeros(10)
    with pytest.raises(ValueError):
        f = AffFunc.from_mats(mat, bias)


def test_constructor_shape_bias():
    mat = np.zeros((10, 12))
    bias = np.zeros((10, 2))
    with pytest.raises(ValueError):
        f = AffFunc.from_mats(mat, bias)


def test_constructor_id():
    f = AffFunc.identity(5)
    assert f.mat.shape == (5, 5)
    assert f.bias.shape == (5,)


def test_add_affine():
    f = AffFunc.from_mats(np.eye(3), np.array([-1,-2,-3]))
    g = AffFunc.from_mats(np.array([1,2,3,4,5,6,7,8,9]).reshape((3, 3)), np.array([3,6,9]))

    h = AffFunc.add(f, g)

    assert np.allclose(h.mat, np.array([2,2,3,4,6,6,7,8,10]).reshape((3, 3)))
    assert np.allclose(h.bias, np.array([2,4,6]))


def test_apply_affine():
    f = AffFunc.from_mats(np.array([1,2,3,4,5,6,7,8,9]).reshape((3, 3)), np.array([3,6,9]))

    h = f.apply(np.array([1, 0, -1]))

    assert np.allclose(h, np.array([1, 4, 7]))


def test_getitem_affine():
    f = AffFunc.from_mats(np.array([[2, 1], [1, 1]], dtype=np.float), np.array([-1, 0], dtype=np.float))

    assert np.allclose(f[0].mat, np.array([[2, 1]], dtype=np.float))
    assert np.allclose(f[0].bias, np.array([-1], dtype=np.float))


@pytest.mark.skip(reason='Feature not yet implemented')
def test_getitem_slice_affine():
    f = AffFunc.from_mats(np.array([[2, 1], [1, -3], [5, 7]], dtype=np.float), np.array([-1, 0, -5], dtype=np.float))
    g = AffFunc.from_mats(np.array([[1, -3],[5,7]], dtype=np.float), np.array([0,-5], dtype=np.float))

    assert f[1:3] == g


@pytest.mark.skip(reason='Feature not yet implemented')
def test_polyhedra():
    A = np.array([[1,0],[-1, 1]])
    b = np.array([-2,3])

    rv = Polytope.hrepr('Ax+b<=0')
    gt = Polytope.from_Axbleqz(A, b)
    assert rv == gt
    rv = Polytope.hrepr('Ax+b>=0')
    gt = Polytope.from_Axbgeqz(A, b)
    assert rv == gt
    rv = Polytope.hrepr('Ax<=b')
    gt = Polytope.from_Axleqb(A, b)
    assert rv == gt
    rv = Polytope.hrepr('Ax>=b')
    gt = Polytope.from_Axgeqb(A, b)
    assert rv == gt


def test_polyhedra_contains_triangle():
    A = np.array([[1, 1], [-1, 1], [0, -1]])
    b = np.array([0, 0, 2.4])

    p = Polytope.from_mats(A, b)

    # assert np.array([0, -1.4]) in p
    assert p.contains(np.array([0, -1.4]))


def test_chebyshev_center_box():
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([1, 1, 1, 1])

    p = Polytope.from_mats(A, b)
    cpoly, cost = p.chebyshev_center()
    res = cpoly.solve(cost)
    
    assert np.allclose(res[:-1], np.array([0, 0]))
    assert np.allclose(res[-1], np.array([1]))


def test_chebyshev_center_box2():
    # construct rectangle (-1,-1) -| (2,1)
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([2, 1, 1, 1])

    p = Polytope.from_mats(A, b)
    cpoly, cost = p.chebyshev_center()
    res = cpoly.solve(cost)
    
    assert 0.0 <= res[0] <= 1.0
    assert res[1] == 0.0
    assert res[2] == 1.0


def test_chebyshev_center_triangle():
    A = np.array([[1, 1], [-1, 1], [0, -1]])
    b = np.array([0, 0, 2.4])

    p = Polytope.from_mats(A, b)
    cpoly, cost = p.chebyshev_center()
    res = cpoly.solve(cost)

    assert np.allclose(res[:-1], np.array([0, -1.414]), atol=1e-2)
    assert np.allclose(res[-1], np.array([1]), atol=1e-2)


def test_chebyshev_center_unbound_triangle():
    A = np.array([[1, 1], [-1, 1]])
    b = np.array([0, 0])

    p = Polytope.from_mats(A, b)
    cpoly, cost = p.chebyshev_center()
    res = cpoly.solve(cost)

    assert np.isnan(res[0])
    assert np.isnan(res[1])
    assert np.isinf(res[2])
        

