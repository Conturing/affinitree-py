#   Copyright 2025 affinitree developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import pytest

from affinitree import AffFunc, Polytope


def test_from_mats():
    mat = np.array([[2, 0], [0, 3]])
    bias = np.array([1, 1])
    func = AffFunc.from_mats(mat, bias)

    assert np.allclose(func(np.array([2, 1])), np.array([5, 4]))


def test_constructor_row_mismatch():
    mat = np.zeros((10, 12))
    bias = np.zeros(12)
    with pytest.raises(Exception):
        func = AffFunc.from_mats(mat, bias)


def test_constructor_shape_mat():
    mat = np.zeros((10, 12, 2))
    bias = np.zeros(10)
    with pytest.raises(ValueError):
        func = AffFunc.from_mats(mat, bias)


def test_constructor_shape_bias():
    mat = np.zeros((10, 12))
    bias = np.zeros((10, 2))
    with pytest.raises(ValueError):
        func = AffFunc.from_mats(mat, bias)


def test_identity_eval():
    func = AffFunc.identity(3)
    input_vector = np.array([3, 5, 7])
    assert np.allclose(func(input_vector), input_vector)


def test_identity_shape():
    func = AffFunc.identity(5)
    assert func.mat.shape == (5, 5)
    assert func.bias.shape == (5,)


def test_zeros():
    # should return zero vector
    func = AffFunc.zeros(4)
    assert np.array_equal(func(np.array([1, 2, 3, 4])), np.zeros(4))


def test_constant():
    # should return constant value as output
    func = AffFunc.constant(5, -8)
    assert np.array_equal(func(np.array([1, 2, 3, 4, 5])), np.array([-8]))


def test_unit():
    # should return the element at index 1
    func = AffFunc.unit(3, 1)
    assert np.array_equal(func(np.array([1, 2, 3])), np.array([2]))


def test_zero_idx():
    # should set index 2 to zero
    func = AffFunc.zero_idx(3, 2)
    assert np.array_equal(func(np.array([1, 2, 3])), np.array([1, 2, 0]) )


def test_rotation():
    # Test rotation function with a 90 degree rotation matrix
    orthogonal_mat = np.array([[0, -1], [1, 0]])
    func = AffFunc.rotation(orthogonal_mat)
    assert np.array_equal(func(np.array([1, 0])), np.array([0, 1]))


def test_uniform_scaling():
    func = AffFunc.uniform_scaling(2, 3.0)
    assert np.array_equal(func(np.array([3, -4])), np.array([9, -12]))


def test_scaling():
    func = AffFunc.scaling(np.array([2, 3]))
    assert np.array_equal(func(np.array([3, -4])), np.array([6, -12]))


@pytest.mark.skip(reason="requires fix in rust version")
def test_translation():
    func = AffFunc.translation(2, np.array([2, 3]))
    assert np.array_equal(func(np.array([4, -2])), np.array([6, 1]))


def test_apply():
    func = AffFunc.from_mats(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((3, 3)), np.array([3, 6, 9]))

    assert np.allclose(func.apply(np.array([1, 0, -1])), np.array([1, 4, 7]))


def test_apply_transpose():
    mat = np.array([[1, 2], [3, 4]])
    bias = np.array([2, 1])
    func = AffFunc.from_mats(mat, bias)
    assert np.array_equal(func.apply_transpose(np.array([2, 3])), np.array([6, 8]))


def test_compose():
    mat1 = np.array([[1, 2], [3, 4]])
    bias1 = np.array([-3, 1])
    func1 = AffFunc.from_mats(mat1, bias1)
    
    mat2 = np.array([[2, 0], [1, 2]])
    bias2 = np.array([0, -1])
    func2 = AffFunc.from_mats(mat2, bias2)

    composed_func = func1.compose(func2)
    
    assert np.array_equal(composed_func.apply(np.array([4, -7])), func1.apply(func2.apply(np.array([4, -7]))))


def test_stack():
    mat1 = np.array([[1, 0]])
    bias1 = np.array([2])
    func1 = AffFunc.from_mats(mat1, bias1)

    mat2 = np.array([[0, 1]])
    bias2 = np.array([-3])
    func2 = AffFunc.from_mats(mat2, bias2)

    stacked_func = func1.stack(func2)
    
    assert np.array_equal(stacked_func.apply(np.array([2, 5])), np.array([4, 2]))


def test_add_affine():
    f = AffFunc.from_mats(np.eye(3), np.array([-1,-2,-3]))
    g = AffFunc.from_mats(np.array([1,2,3,4,5,6,7,8,9]).reshape((3, 3)), np.array([3,6,9]))

    h = f + g

    assert np.allclose(h.mat, np.array([2,2,3,4,6,6,7,8,10]).reshape((3, 3)))
    assert np.allclose(h.bias, np.array([2,4,6]))


def test_getitem_affine():
    f = AffFunc.from_mats(np.array([[2, 1], [1, 1]], dtype=np.float64), np.array([-1, 0], dtype=np.float64))

    assert np.allclose(f[0].mat, np.array([[2, 1]], dtype=np.float64))
    assert np.allclose(f[0].bias, np.array([-1], dtype=np.float64))


def test_polyhedra_contains_triangle():
    A = np.array([[1, 1], [-1, 1], [0, -1]])
    b = np.array([0, 0, 2.4])

    p = Polytope.from_mats(A, b)

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


def test_chebyshev_center_unbounded_triangle():
    A = np.array([[1, 1], [-1, 1]])
    b = np.array([0, 0])

    p = Polytope.from_mats(A, b)
    cpoly, cost = p.chebyshev_center()
    
    with pytest.raises(Exception):
        res = cpoly.solve(cost)

    # assert np.isnan(res[0])
    # assert np.isnan(res[1])
    # assert np.isinf(res[2])
        

