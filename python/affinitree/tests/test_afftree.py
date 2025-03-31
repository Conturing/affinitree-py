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
import torch
from torch import nn

from affinitree import AffTree, AffFunc, Polytope
from affinitree import schema


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
    dd.compose(schema.partial_ReLU(2, 0))
    dd.compose(schema.partial_ReLU(2, 1))

    assert np.allclose(dd.evaluate(np.array([6., -7.])), np.array([4., 0.]))


def test_root():
    f = AffFunc.from_mats(np.array([[1., -3., 2], [0., 1., -5.], [-2., 3., 6.]]), np.array([2., -4., -5.]))
    dd = AffTree.from_aff(f)
    
    assert np.allclose(dd.root.val.mat, f.mat)
    assert np.allclose(dd.root.val.bias, f.bias)
    
    # turn root from terminal into decision node
    dd.compose(schema.partial_ReLU(3, 0))
    dd.compose(schema.partial_ReLU(3, 1))
    dd.compose(schema.partial_ReLU(3, 2))
    dd.apply_func(f)
    dd.compose(schema.partial_ReLU(3, 0))
    dd.compose(schema.partial_ReLU(3, 1))
    dd.compose(schema.partial_ReLU(3, 2))
    
    assert np.allclose(dd.root.val.mat, np.array([[1., -3., 2]]))
    assert np.allclose(dd.root.val.bias, np.array([-2.]))
    
    
def test_size():
    f = AffFunc.from_mats(np.array([[1., -3., 2], [0., 1., -5.], [-2., 3., 6.]]), np.array([2., -4., -5.]))
    dd = AffTree.from_aff(f)
    assert dd.size() == 1
    
    dd.compose(schema.partial_ReLU(3, 0), prune=False)
    dd.compose(schema.partial_ReLU(3, 1), prune=False)
    dd.compose(schema.partial_ReLU(3, 2), prune=False)
    assert dd.size() == 15
    
    dd.apply_func(f)
    assert dd.size() == 15
    
    dd.compose(schema.partial_ReLU(3, 0), prune=False)
    dd.compose(schema.partial_ReLU(3, 1), prune=False)
    dd.compose(schema.partial_ReLU(3, 2), prune=False)
    assert dd.size() == 127
    
    
def test_depth():
    f = AffFunc.from_mats(np.array([[1., -3., 2], [0., 1., -5.], [-2., 3., 6.]]), np.array([2., -4., -5.]))
    dd = AffTree.from_aff(f)
    assert dd.depth() == 0
    
    dd.compose(schema.partial_ReLU(3, 0), prune=False)
    dd.compose(schema.partial_ReLU(3, 1), prune=False)
    dd.compose(schema.partial_ReLU(3, 2), prune=False)
    assert dd.depth() == 3
    
    dd.apply_func(f)
    assert dd.depth() == 3
    
    dd.compose(schema.partial_ReLU(3, 0), prune=False)
    dd.compose(schema.partial_ReLU(3, 1), prune=False)
    dd.compose(schema.partial_ReLU(3, 2), prune=False)
    assert dd.depth() == 6


def test_indim():
    f = AffFunc.from_mats(np.array([[1., -3., 2], [0., 1., -5.], [-2., 3., 6.]]), np.array([2., -4., -5.]))
    dd = AffTree.from_aff(f)
    
    assert dd.indim() == 3
    
    dd.compose(schema.partial_ReLU(3, 0))
    dd.compose(schema.partial_ReLU(3, 1))
    dd.compose(schema.partial_ReLU(3, 2))
    dd.apply_func(f)
    dd.compose(schema.partial_ReLU(3, 0))
    dd.compose(schema.partial_ReLU(3, 1))
    dd.compose(schema.partial_ReLU(3, 2))
    
    assert dd.indim() == 3


def test_polyhedra():
    dd = AffTree.from_aff(AffFunc.from_mats(np.array([[1., 2.], [2., 1.], [-1., 3.], [9., -4.]]), np.array([0., 2., 3., -5.])))
    dd.compose(schema.partial_ReLU(4, 0), prune=False)
    dd.compose(schema.partial_ReLU(4, 1), prune=False)
    dd.compose(schema.partial_ReLU(4, 2), prune=False)
    dd.compose(schema.partial_ReLU(4, 3), prune=False)
    
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
    dd.compose(schema.partial_ReLU(2, 0))
    dd.compose(schema.partial_ReLU(2, 1))
    dd.apply_func(g)
    dd.compose(schema.partial_ReLU(2, 0))
    dd.compose(schema.partial_ReLU(2, 1))
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
    dd.compose(schema.partial_ReLU(2, 0))
    dd.compose(schema.partial_ReLU(2, 1))
    dd.apply_func(g)
    dd.compose(schema.partial_ReLU(2, 0))
    dd.compose(schema.partial_ReLU(2, 1))
    dd.apply_func(h)

    def affine_to_layer(a: AffFunc) -> nn.Linear:
        layer = nn.Linear(a.indim(), a.outdim())
        layer.weight.data = torch.from_numpy(a.mat)
        layer.bias.data = torch.from_numpy(a.bias)
        return layer

    modules = [affine_to_layer(f), nn.ReLU(), affine_to_layer(g), nn.ReLU(), affine_to_layer(h)]
    net = nn.Sequential(*modules)

    assert_equiv_net(net, dd)


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
n0 [label="+1.00 $0 +0.50 $1 ≤ +0.50", shape=ellipse];
n1 [label="+1.00 $0 +1.00 $1 ≤ +0.00", shape=ellipse];
n2 [label="+1.00 $0 +1.00 $1 ≤ +0.00", shape=ellipse];
n3 [label="+1.00 $0 +0.50 $1 ≤ −0.50", shape=ellipse];
n4 [label="+1.00 $0 +0.50 $1 ≤ −0.50", shape=ellipse];
n5 [label="⊥", shape=ellipse];
n6 [label="⊥", shape=ellipse];
n7 [label="+1.00 $0 +0.80 $1 ≤ +0.20", shape=ellipse];
n8 [label="+1.00 $0 +0.80 $1 ≤ +0.20", shape=ellipse];
n9 [label="+1.00 $0 +0.50 $1 ≤ +0.50", shape=ellipse];
n10 [label="+1.00 $0 +0.50 $1 ≤ +0.50", shape=ellipse];
n11 [label="+1.00 $0 +1.00 $1 ≤ +0.00", shape=ellipse];
n12 [label="+1.00 $0 +1.00 $1 ≤ +0.00", shape=ellipse];
n13 [label="⊤", shape=ellipse];
n14 [label="⊤", shape=ellipse];
n15 [label="+1.00 +2.00 $0 +1.00 $1
−1.00 +5.00 $0 +4.00 $1", shape=box];
n16 [label="+1.00 +2.00 $0 +1.00 $1
+0.00 ", shape=box];
n17 [label="+0.00 
−1.00 +5.00 $0 +4.00 $1", shape=box];
n18 [label="+0.00 
+0.00 ", shape=box];
n19 [label="+1.00 +2.00 $0 +1.00 $1
−1.00 +2.00 $0 +1.00 $1", shape=box];
n20 [label="+1.00 +2.00 $0 +1.00 $1
+0.00 ", shape=box];
n21 [label="+0.00 
−1.00 +2.00 $0 +1.00 $1", shape=box];
n22 [label="+0.00 
+0.00 ", shape=box];
n23 [label="+2.00 
+0.00 +3.00 $0 +3.00 $1", shape=box];
n24 [label="+2.00 
+0.00 ", shape=box];
n25 [label="+0.00 
+0.00 +3.00 $0 +3.00 $1", shape=box];
n26 [label="+0.00 
+0.00 ", shape=box];
n27 [label="+2.00 
+0.00 ", shape=box];
n28 [label="+2.00 
+0.00 ", shape=box];
n29 [label="+0.00 
+0.00 ", shape=box];
n30 [label="+0.00 
+0.00 ", shape=box];
n0 -> n1 [label=0, style=dashed];
n0 -> n2 [label=1, style=solid];
n1 -> n3 [label=0, style=dashed];
n1 -> n4 [label=1, style=solid];
n2 -> n5 [label=0, style=dashed];
n2 -> n6 [label=1, style=solid];
n3 -> n7 [label=0, style=dashed];
n3 -> n8 [label=1, style=solid];
n4 -> n9 [label=0, style=dashed];
n4 -> n10 [label=1, style=solid];
n5 -> n11 [label=0, style=dashed];
n5 -> n12 [label=1, style=solid];
n6 -> n13 [label=0, style=dashed];
n6 -> n14 [label=1, style=solid];
n7 -> n15 [label=0, style=dashed];
n7 -> n16 [label=1, style=solid];
n8 -> n17 [label=0, style=dashed];
n8 -> n18 [label=1, style=solid];
n9 -> n19 [label=0, style=dashed];
n9 -> n20 [label=1, style=solid];
n10 -> n21 [label=0, style=dashed];
n10 -> n22 [label=1, style=solid];
n11 -> n23 [label=0, style=dashed];
n11 -> n24 [label=1, style=solid];
n12 -> n25 [label=0, style=dashed];
n12 -> n26 [label=1, style=solid];
n13 -> n27 [label=0, style=dashed];
n13 -> n28 [label=1, style=solid];
n14 -> n29 [label=0, style=dashed];
n14 -> n30 [label=1, style=solid];
}'''

def test_dot_str():
    dd = AffTree.identity(2)
    f = AffFunc.from_mats(np.array([[2, 1], [1, 1]], dtype=np.float64), np.array([-1, 0], dtype=np.float64))
    g = AffFunc.from_mats(np.array([[1, 0], [1, 3]], dtype=np.float64), np.array([2, 0], dtype=np.float64))

    dd.apply_func(f)
    dd.compose(schema.partial_ReLU(2, 0), prune=False)
    dd.compose(schema.partial_ReLU(2, 1), prune=False)
    dd.apply_func(g)
    dd.compose(schema.partial_ReLU(2, 0), prune=False)
    dd.compose(schema.partial_ReLU(2, 1), prune=False)

    print(dd.to_dot())

    assert dd.to_dot() == DOT_STR


def test_operator_add():
    tree0 = AffTree.from_array(np.array([[1, 0]]), np.array([0])) 
    tree1 = AffTree.from_array(np.array([[1, 0]]), np.array([0])) 

    tree = tree0 + tree1
    assert np.allclose(tree.evaluate(np.array([7., 2.])), np.array([14.]))


def test_operator_sub():
    tree0 = AffTree.from_array(np.array([[1, 0]]), np.array([0])) 
    tree1 = AffTree.from_array(np.array([[1, 0]]), np.array([0])) 

    tree = tree0 - tree1
    assert np.allclose(tree.evaluate(np.array([7., 2.])), np.array([0.]))


def test_operator_getitem():
    tree0 = AffTree.from_array(np.array([[1, 0]]), np.array([0])) 

    root = tree0[0]
    assert root.id == tree0.root.id


@pytest.fixture()
def complex_afftree():
    tree = AffTree.from_array(np.array([[1, 0]]), np.array([0]))    # x <= 0
    
    pred1 = AffFunc.from_mats(np.array([[1, 0]]), np.array([2]))    # x <= 2
    pred2 = AffFunc.from_mats(np.array([[0, 1]]), np.array([0]))    # y <= 0
    pred3 = AffFunc.from_mats(np.array([[1, 1]]), np.array([1]))    # x + y <= 1
    func4 = AffFunc.from_mats(np.array([[-1, -1]]), np.array([1]))  # f4(x, y) = -x - y + 1
    pred5 = AffFunc.from_mats(np.array([[1, -1]]), np.array([1]))   # x - y <= 1
    func6 = AffFunc.from_mats(np.array([[0, 1]]), np.array([1]))    # f5(x, y) = y + 1
    func7 = AffFunc.from_mats(np.array([[2, 2]]), np.array([0]))    # f7(x, y) = 2x + 2y
    func8 = AffFunc.from_mats(np.array([[2, 2]]), np.array([0]))    # f8(x, y) = 2x + 2y
    func9 = AffFunc.from_mats(np.array([[1, 0]]), np.array([-2]))   # f9(x, y) = x - 2
    func10 = AffFunc.from_mats(np.array([[0, -1]]), np.array([-2])) # f10(x, y) = -y - 2

    left_node = tree.add_child_node(tree.root, 0, pred1)
    right_node = tree.add_child_node(tree.root, 1, pred2)
    
    ll_node = tree.add_child_node(left_node, 0, pred3)
    tree.add_child_node(left_node, 1, func4)
    tree.add_child_node(right_node, 0, pred5)
    tree.add_child_node(right_node, 1, func6)
    
    tree.add_child_node(ll_node, 0, func7)
    tree.add_child_node(ll_node, 1, func8)
    tree.add_child_node(tree.child(right_node, 0), 0, func9)
    tree.add_child_node(tree.child(right_node, 0), 1, func10)
    
    return tree


def test_depth2(complex_afftree: AffTree):
    assert complex_afftree.depth() == 3


def test_size2(complex_afftree: AffTree):
    assert complex_afftree.size() == 11


def test_infeasible2(complex_afftree: AffTree):
    complex_afftree.infeasible_elimination()
    assert complex_afftree.depth() == 3
    assert complex_afftree.size() == 9


def test_reduce2(complex_afftree: AffTree):
    complex_afftree.reduce()
    assert complex_afftree.depth() == 3
    assert complex_afftree.size() == 9


@pytest.fixture()
def afftree_max_relu():
    # tree of the function max(relu(x), relu(y))
    tree = AffTree.from_array(np.array([[1, 0]]), np.array([0]))    # x <= 0
    
    pred1 = AffFunc.from_mats(np.array([[0, 1]]), np.array([0]))    # y <= 0
    pred2 = AffFunc.from_mats(np.array([[0, 1]]), np.array([0]))    # y <= 0
    pred3 = AffFunc.from_mats(np.array([[1, -1]]), np.array([0]))   # x <= y
    pred4 = AffFunc.from_mats(np.array([[1, 0]]), np.array([0]))    # x <= 0
    pred5 = AffFunc.from_mats(np.array([[1, 0]]), np.array([0]))    # y <= 0
    pred6 = AffFunc.from_mats(np.array([[0, 0]]), np.array([0]))    # 0 <= 0

    func1 = AffFunc.from_mats(np.array([[0, 0]]), np.array([0]))    # 0
    func2 = AffFunc.from_mats(np.array([[0, 0]]), np.array([0]))    # 0
    func3 = AffFunc.from_mats(np.array([[0, 0]]), np.array([0]))    # 0
    func4 = AffFunc.from_mats(np.array([[0, 1]]), np.array([0]))    # y
    func5 = AffFunc.from_mats(np.array([[0, 0]]), np.array([0]))    # 0
    func6 = AffFunc.from_mats(np.array([[1, 0]]), np.array([0]))    # x
    func7 = AffFunc.from_mats(np.array([[0, 1]]), np.array([0]))    # y
    func8 = AffFunc.from_mats(np.array([[1, 0]]), np.array([0]))    # x

    left_node = tree.add_child_node(tree.root, 0, pred1)
    right_node = tree.add_child_node(tree.root, 1, pred2)
    
    ll_node = tree.add_child_node(left_node, 0, pred3)
    rl_node = tree.add_child_node(left_node, 1, pred4)
    lr_node = tree.add_child_node(right_node, 0, pred5)
    rr_node = tree.add_child_node(right_node, 1, pred6)
    
    tree.add_child_node(ll_node, 0, func8)
    tree.add_child_node(ll_node, 1, func7)
    tree.add_child_node(rl_node, 0, func6)
    tree.add_child_node(rl_node, 1, func5)
    tree.add_child_node(lr_node, 0, func3)
    tree.add_child_node(lr_node, 1, func4)
    tree.add_child_node(rr_node, 0, func1)
    tree.add_child_node(rr_node, 1, func2)
    
    return tree


def test_eval(afftree_max_relu: AffTree):
    print(afftree_max_relu.to_dot())
    assert afftree_max_relu.evaluate(np.array([-1, -2]))[0] == 0
    assert afftree_max_relu.evaluate(np.array([-1, 2]))[0] == 2
    assert afftree_max_relu.evaluate(np.array([1, -2]))[0] == 1
    assert afftree_max_relu.evaluate(np.array([1, 2]))[0] == 2
    assert afftree_max_relu.evaluate(np.array([3, 2]))[0] == 3
    