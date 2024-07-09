#   Copyright 2024 affinitree developers
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
import torch
from torch import nn

from affinitree import AffTree, AffFunc, Polytope
from affinitree import schema


def assert_equiv_trees(dd0, dd1):
    torch.manual_seed(42)
    rnd = np.random.default_rng(42)

    for idx in range(500):
        x = 100 * rnd.random(2, dtype=float) - 20

        dd1_out = dd1.evaluate(x)
        dd0_out = dd0.evaluate(x)

        assert np.allclose(dd1_out, dd0_out, atol=1e-05)


def assert_equiv_net(model, dd):
    torch.manual_seed(42)
    rnd = np.random.default_rng(42)

    for idx in range(500):
        x = 100 * rnd.random(2, dtype=float) - 20

        net_out = model.forward(torch.from_numpy(x))
        dd_out = dd.evaluate(x)

        assert torch.allclose(net_out, torch.from_numpy(dd_out), atol=1e-05)

#####

def test_identity_constructor():
    dd = AffTree.identity(2)

    assert dd.size() == 1
    assert dd.indim() == 2
    assert np.allclose(dd.evaluate(np.array([6., -7.])), np.array([6., -7.]))


def test_precondition_constructor():
    precondition = Polytope.hyperrectangle(5, [(-1, 1)] * 5)
    dd = AffTree.from_poly(precondition, AffFunc.identity(5))
    
    assert np.allclose(dd.evaluate(np.array([0.5, -0.3, 0.9, 0.2, -0.7])), np.array([0.5, -0.3, 0.9, 0.2, -0.7]))
    
    with pytest.raises(BaseException):
        dd.evaluate(np.array([0.5, -0.3, 1.9, 0.2, -0.7]))
        

def test_apply_func():
    dd = AffTree.identity(2)
    f = AffFunc.from_mats(np.array([[2, 1], [1, 1]]), np.array([-1, 0]))

    dd.apply_func(f)

    assert np.allclose(dd.evaluate(np.array([6., -7.])), np.array([4., -1.]))


def test_evaluate_relu():
    dd = AffTree.identity(2)
    f = AffFunc.from_mats(np.array([[2, 1], [1, 1]]), np.array([-1, 0]))

    dd.apply_func(f)
    dd.compose(schema.ReLU(2))

    assert np.allclose(dd.evaluate(np.array([6., -7.])), np.array([4., 0.]))


def test_root():
    f = AffFunc.from_mats(np.array([[1., -3., 2], [0., 1., -5.], [-2., 3., 6.]]), np.array([2., -4., -5.]))
    dd = AffTree.from_aff(f)
    
    assert np.allclose(dd.root.val.mat, f.mat)
    assert np.allclose(dd.root.val.bias, f.bias)
    
    # turn root from terminal into decision node
    dd.compose(schema.ReLU(3))
    dd.apply_func(f)
    dd.compose(schema.ReLU(3))
    
    assert np.allclose(dd.root.val.mat, np.array([[1., -3., 2]]))
    assert np.allclose(dd.root.val.bias, np.array([2.]))
    
    
def test_size():
    f = AffFunc.from_mats(np.array([[1., -3., 2], [0., 1., -5.], [-2., 3., 6.]]), np.array([2., -4., -5.]))
    dd = AffTree.from_aff(f)
    assert dd.size() == 1
    
    dd.compose(schema.ReLU(3), prune=False)
    assert dd.size() == 15
    
    dd.apply_func(f)
    assert dd.size() == 15
    
    dd.compose(schema.ReLU(3), prune=False)
    assert dd.size() == 127
    
    
def test_depth():
    f = AffFunc.from_mats(np.array([[1., -3., 2], [0., 1., -5.], [-2., 3., 6.]]), np.array([2., -4., -5.]))
    dd = AffTree.from_aff(f)
    assert dd.depth() == 0
    
    dd.compose(schema.ReLU(3), prune=False)
    assert dd.depth() == 3
    
    dd.apply_func(f)
    assert dd.depth() == 3
    
    dd.compose(schema.ReLU(3), prune=False)
    assert dd.depth() == 6


def test_indim():
    f = AffFunc.from_mats(np.array([[1., -3., 2], [0., 1., -5.], [-2., 3., 6.]]), np.array([2., -4., -5.]))
    dd = AffTree.from_aff(f)
    
    assert dd.indim() == 3
    
    dd.compose(schema.ReLU(3))
    dd.apply_func(f)
    dd.compose(schema.ReLU(3))
    
    assert dd.indim() == 3


def test_polyhedra():
    dd = AffTree.from_aff(AffFunc.from_mats(np.array([[1., 2.], [2., 1.], [-1., 3.], [9., -4.]]), np.array([0., 2., 3., -5.])))
    dd.compose(schema.ReLU(4), prune=False)
    
    poly = dd.polyhedra()
    
    assert len(poly) == 31


def test_remove_axes():
    dd = AffTree.identity(6)
    dd.remove_axes(np.array([False, True, False, True, False, False]))
    
    assert dd.indim() == 2
    assert np.allclose(dd.root.val.mat, np.array([[0., 0.], [1., 0.], [0., 0.], [0., 1.], [0., 0.], [0., 0.]]))


def test_reduce_zero():
    dd = AffTree.identity(2)
    f = AffFunc.from_mats(np.array([[2, 1], [1, 1]], dtype=np.float64), np.array([-1, 0], dtype=np.float64))
    g = AffFunc.from_mats(np.array([[0, 0], [0, 0]], dtype=np.float64), np.array([0, 0], dtype=np.float64))

    dd.apply_func(f)
    dd.compose(schema.ReLU(2))
    dd.apply_func(g)
    dd.compose(schema.ReLU(2))
    dd.reduce()

    print(dd.to_dot())

    assert np.allclose(dd[0].val.mat, np.array([[2., 1.]]))
    assert dd.size() == 3


def test_net_equiv():
    dd = AffTree.identity(2)
    f = AffFunc.from_mats(np.array([[2., 1.], [1., 1.]]), np.array([-1., 0.]))
    g = AffFunc.from_mats(np.array([[1., 0.], [1., 3.]]), np.array([2., 0.]))
    h = AffFunc.from_mats(np.array([[2., 3.], [-2., 3.], [1., 0.]]),
                       np.array([2., 0., 1.]))

    dd.apply_func(f)
    dd.compose(schema.ReLU(2))
    dd.apply_func(g)
    dd.compose(schema.ReLU(2))
    dd.apply_func(h)

    def affine_to_layer(a: AffFunc) -> nn.Linear:
        layer = nn.Linear(a.indim(), a.outdim())
        layer.weight.data = torch.from_numpy(a.mat)
        layer.bias.data = torch.from_numpy(a.bias)
        return layer

    modules = [affine_to_layer(f), nn.ReLU(), affine_to_layer(g), nn.ReLU(), affine_to_layer(h)]
    net = nn.Sequential(*modules)

    assert_equiv_net(net, dd)


def test_infeasible_multiple_labels():
    add = AffTree.identity(2)
    f = AffFunc.from_mats(np.array([[2, 1], [1, -2], [0, 1]], dtype=np.float64), np.array([-1, 0, 5], dtype=np.float64))
    g = AffFunc.from_mats(np.array([[1, 0, 2], [1, 3, 1], [1, 0, 0]], dtype=np.float64), np.array([2, 0, -1], dtype=np.float64))

    add.apply_func(f)
    add.compose(schema.partial_ReLU(3, 0))
    add.compose(schema.partial_ReLU(3, 1))
    add.compose(schema.partial_ReLU(3, 2))
    add.apply_func(g)
    add.compose(schema.partial_ReLU(3, 0))
    add.compose(schema.partial_ReLU(3, 1))
    add.compose(schema.partial_ReLU(3, 2))

    add.infeasible_elimination()

    cdd = AffTree.identity(2)
    cdd.apply_func(f)
    cdd.compose(schema.ReLU(3))
    cdd.apply_func(g)
    cdd.compose(schema.ReLU(3))

    assert_equiv_trees(add, cdd)


def test_argmax():
    dd = AffTree.identity(1)
    dd.apply_func(AffFunc.from_mats(np.array([[2], [4], [8]]), np.array([0, -2, -4])))

    dd.compose(schema.argmax(3))

    res = dd.evaluate(np.array([6], dtype=np.float64))
    assert res[0] == 2
    
    res = dd.evaluate(np.array([-2], dtype=np.float64))
    assert res[0] == 0

DOT_STR = '''digraph afftree {
bgcolor=transparent;
concentrate=true;
margin=0;
n0 [label="−1.00 $0 −0.50 $1 ≤ −0.50", shape=ellipse];
n1 [label="−1.00 $0 −1.00 $1 ≤ +0.00", shape=ellipse];
n2 [label="−1.00 $0 −1.00 $1 ≤ +0.00", shape=ellipse];
n3 [label="−1.00 $0 −0.50 $1 ≤ +0.50", shape=ellipse];
n4 [label="−1.00 $0 −0.50 $1 ≤ +0.50", shape=ellipse];
n5 [label="⊤", shape=ellipse];
n6 [label="⊤", shape=ellipse];
n7 [label="−1.00 $0 −0.50 $1 ≤ −0.50", shape=ellipse];
n8 [label="−1.00 $0 −0.50 $1 ≤ −0.50", shape=ellipse];
n9 [label="+1.00 +2.00 $0 +1.00 $1
+0.00 ", shape=box];
n10 [label="+1.00 +2.00 $0 +1.00 $1
−1.00 +2.00 $0 +1.00 $1", shape=box];
n11 [label="+0.00 
+0.00 ", shape=box];
n12 [label="+0.00 
−1.00 +2.00 $0 +1.00 $1", shape=box];
n13 [label="−1.00 $0 −0.80 $1 ≤ −0.20", shape=ellipse];
n14 [label="−1.00 $0 −0.80 $1 ≤ −0.20", shape=ellipse];
n15 [label="+1.00 +2.00 $0 +1.00 $1
+0.00 ", shape=box];
n16 [label="+1.00 +2.00 $0 +1.00 $1
−1.00 +5.00 $0 +4.00 $1", shape=box];
n17 [label="+0.00 
+0.00 ", shape=box];
n18 [label="+0.00 
−1.00 +5.00 $0 +4.00 $1", shape=box];
n19 [label="⊤", shape=ellipse];
n20 [label="⊤", shape=ellipse];
n21 [label="+2.00 
+0.00 ", shape=box];
n22 [label="+2.00 
+0.00 ", shape=box];
n23 [label="+0.00 
+0.00 ", shape=box];
n24 [label="+0.00 
+0.00 ", shape=box];
n25 [label="−1.00 $0 −1.00 $1 ≤ +0.00", shape=ellipse];
n26 [label="−1.00 $0 −1.00 $1 ≤ +0.00", shape=ellipse];
n27 [label="+2.00 
+0.00 ", shape=box];
n28 [label="+2.00 
+0.00 +3.00 $0 +3.00 $1", shape=box];
n29 [label="+0.00 
+0.00 ", shape=box];
n30 [label="+0.00 
+0.00 +3.00 $0 +3.00 $1", shape=box];
n0 -> n1 [label=0, style=dashed];
n0 -> n2 [label=1, style=solid];
n2 -> n3 [label=0, style=dashed];
n2 -> n4 [label=1, style=solid];
n1 -> n5 [label=0, style=dashed];
n1 -> n6 [label=1, style=solid];
n3 -> n7 [label=0, style=dashed];
n3 -> n8 [label=1, style=solid];
n8 -> n9 [label=0, style=dashed];
n8 -> n10 [label=1, style=solid];
n7 -> n11 [label=0, style=dashed];
n7 -> n12 [label=1, style=solid];
n4 -> n13 [label=0, style=dashed];
n4 -> n14 [label=1, style=solid];
n14 -> n15 [label=0, style=dashed];
n14 -> n16 [label=1, style=solid];
n13 -> n17 [label=0, style=dashed];
n13 -> n18 [label=1, style=solid];
n5 -> n19 [label=0, style=dashed];
n5 -> n20 [label=1, style=solid];
n20 -> n21 [label=0, style=dashed];
n20 -> n22 [label=1, style=solid];
n19 -> n23 [label=0, style=dashed];
n19 -> n24 [label=1, style=solid];
n6 -> n25 [label=0, style=dashed];
n6 -> n26 [label=1, style=solid];
n26 -> n27 [label=0, style=dashed];
n26 -> n28 [label=1, style=solid];
n25 -> n29 [label=0, style=dashed];
n25 -> n30 [label=1, style=solid];
}'''

def test_dot_str():
    dd = AffTree.identity(2)
    f = AffFunc.from_mats(np.array([[2, 1], [1, 1]], dtype=np.float64), np.array([-1, 0], dtype=np.float64))
    g = AffFunc.from_mats(np.array([[1, 0], [1, 3]], dtype=np.float64), np.array([2, 0], dtype=np.float64))

    dd.apply_func(f)
    dd.compose(schema.ReLU(2), prune=False)
    dd.apply_func(g)
    dd.compose(schema.ReLU(2), prune=False)

    print(dd.to_dot())

    assert dd.to_dot() == DOT_STR