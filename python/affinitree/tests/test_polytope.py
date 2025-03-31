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

def test_from_mats_valid():
    # should create a Polytope with given matrix and bias
    mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    bias = np.array([1, 0, 1, 0])
    poly = Polytope.from_mats(mat, bias)
    assert np.array_equal(poly.mat, mat)
    assert np.array_equal(poly.bias, bias)


def test_from_mats_invalid_shape():
    # should raise an error for mismatched shapes
    mat = np.array([[1, 0], [-1, 0]])
    bias = np.array([1, 0, 1]) 
    with pytest.raises(ValueError):
        Polytope.from_mats(mat, bias)


def test_hyperrectangle_valid():
    # should create a hyperrectangle with specified intervals
    intervals = [(0, 1), (0, 2)]
    poly = Polytope.hyperrectangle(2, intervals)
    assert poly.contains(np.array([0.2, 1.0]))
    assert poly.contains(np.array([0.8, 0.1]))
    assert not poly.contains(np.array([1.3, 1.0]))


@pytest.mark.skip(reason="PanicException does not inherit from Exception")
def test_hyperrectangle_invalid_intervals():
    # should raise an error for invalid intervals
    intervals = [(1, 0)]
    with pytest.raises(Exception): # in future versions maybe ValueError
        Polytope.hyperrectangle(1, intervals)


def test_indim():
    poly = Polytope.from_mats(np.array([[1, 0], [-1, 0]]), np.array([1, 0]))
    assert poly.indim() == 2

    intervals = [(0, 1), (0, 2), (-4, -2)]
    poly = Polytope.hyperrectangle(3, intervals)
    assert poly.indim() == 3


def test_n_constraints():
    mat = np.array([[1, 0], [-1, 0], [0, 1]])
    bias = np.array([1, 0, 1])
    poly = Polytope.from_mats(mat, bias)
    assert poly.n_constraints() == 3


def test_row_valid():
    # should return a new Polytope from the second row
    mat = np.array([[1, 0], [-1, 0], [0, 1]])
    bias = np.array([1, 0, 1])
    poly = Polytope.from_mats(mat, bias)
    sub_poly = poly.row(1)  
    assert np.array_equal(sub_poly.mat, np.array([[-1, 0]]))
    assert np.array_equal(sub_poly.bias, np.array([0]))


@pytest.mark.skip(reason="PanicException does not inherit from Exception")
def test_row_invalid():
    # should raise an error for invalid row index
    mat = np.array([[1, 0], [-1, 0]])
    bias = np.array([1, 0])
    poly = Polytope.from_mats(mat, bias)
    with pytest.raises(Exception): # in future versions maybe IndexError
        poly.row(2)


def test_row_iter():
    mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    bias = np.array([1, 0, 2, 0])
    poly = Polytope.from_mats(mat, bias)
    
    rows = poly.row_iter()
    assert len(rows) == 4
    assert all(isinstance(row, Polytope) for row in rows)


def test_distance_within_polytope():
    # should return non-negative distances for a point inside the polytope
    mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    bias = np.array([1, 0, 1, 0])
    poly = Polytope.from_mats(mat, bias)
    dist = poly.distance(np.array([0.5, 0.5]))
    assert all(d >= 0 for d in dist)


def test_distance_outside_polytope():
    # should return negative distances for a point outside the polytope
    mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    bias = np.array([1, 0, 1, 0])
    poly = Polytope.from_mats(mat, bias)
    dist = poly.distance(np.array([2, 2]))
    assert any(d < 0 for d in dist)


def test_contains_true():
    # should return True for a point inside the polytope
    mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    bias = np.array([1, 0, 1, 0])
    poly = Polytope.from_mats(mat, bias)
    assert poly.contains(np.array([0.5, 0.5]))


def test_contains_false():
    # should return False for a point outside the polytope
    mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    bias = np.array([1, 0, 1, 0])
    poly = Polytope.from_mats(mat, bias)
    assert not poly.contains(np.array([2, 2]))


def test_normalize():
    # should return a normalized polytope
    mat = np.array([[3, 1], [-2, 4], [-1, -1], [1, 0]])
    bias = np.array([3, 4, 2, 5])
    poly = Polytope.from_mats(mat, bias)
    n_poly = poly.normalize()
    assert np.allclose(np.linalg.norm(n_poly.mat, axis=1), np.ones(4))


def test_translate():
    intervals = [(0, 1), (0, 2)]
    poly = Polytope.hyperrectangle(2, intervals)
    t_poly = poly.translate(np.array([3, 1]))
    
    assert t_poly.contains(np.array([3.5, 1.0]))
    assert not t_poly.contains(np.array([2.5, 1.2])) 


def test_intersection():
    poly0 = Polytope.hyperrectangle(2, [(2, 5), (-1, 2)])
    poly1 = Polytope.hyperrectangle(2, [(0.5, 7), (1, 3)])
    res = poly0.intersection(poly1)
    
    assert res.n_constraints() == 8
    assert res.contains(np.array([3, 1.5]))
    assert not res.contains(np.array([4, 0]))


def test_rotate():
    poly = Polytope.hyperrectangle(2, [(0, 2), (0, 1)])
    r_poly = poly.rotate(np.array([[0, -1], [1, 0]])) # 90-degree rotation
    
    assert r_poly.contains(np.array([-0.5, 1.5])) 
    assert not r_poly.contains(np.array([0.5, 0.5]))


def test_chebyshev_center():
    poly = Polytope.hyperrectangle(2, [(2, 6), (2, 6)])

    lp, coef = poly.chebyshev_center()
    res = lp.solve(coef)
    assert np.allclose(res[:-1], np.array([4.0, 4.0]))  
    assert res[-1] == 2


def test_solve_feasible():
    polytope = Polytope.hyperrectangle(2, [(0, 1), (0, 2)])
    cost = np.array([-1., -1.])
    
    solution = polytope.solve(cost)
    assert np.allclose(solution, np.array([1.0, 2.0]))


def test_solve_infeasible():
    poly = Polytope.from_mats(np.array([[0, 1], [1, 0], [-1, -1]]), np.array([0, 0, -1]))
    with pytest.raises(ValueError):
        poly.solve(np.array([-1, -1]))