import networkx as nx
import numpy as np
import pytest
import torch
from torch import nn

from affinitree import AffTree, AffFunc, Polytope
from affinitree import schema


def assert_equiv_trees(cdd, add):
    torch.manual_seed(42)
    rnd = np.random.default_rng(42)

    for idx in range(500):
        x = 100 * rnd.random(2, dtype=float) - 20

        cdd_out = cdd.evaluate(x)
        add_out = add.evaluate(x)

        assert np.allclose(cdd_out, add_out, atol=1e-05)


def assert_equiv_net(model, add):
    torch.manual_seed(42)
    rnd = np.random.default_rng(42)

    for idx in range(500):
        x = 100 * rnd.random(2, dtype=float) - 20

        net_out = model.forward(torch.from_numpy(x))
        add_out = add.evaluate(x)

        assert torch.allclose(net_out, torch.from_numpy(add_out), atol=1e-05)

#####

def test_identity_constructor():
    dd = AffTree.identity(2)

    assert dd.size() == 1
    assert dd.indim() == 2
    assert np.allclose(dd.evaluate(np.array([6., -7.])), np.array([6., -7.]))


def test_precondition_constructor():
    # precondition = Polytope.hypercube_poly(5, 1)
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


@pytest.mark.skip(reason='Feature not yet implemented')
def test_reduce_zero():
    dd = AffTree.identity(2)
    f = AffFunc.from_mats(np.array([[2, 1], [1, 1]], dtype=np.float), np.array([-1, 0], dtype=np.float))
    g = AffFunc.from_mats(np.array([[0, 0], [0, 0]], dtype=np.float), np.array([0, 0], dtype=np.float))

    dd.apply_func(f)
    dd.compose(schema.ReLU(2))
    dd.apply_func(g)
    dd.compose(schema.ReLU(2))
    dd.reduce()

    assert dd.size() == 2


@pytest.mark.skip(reason='Feature not yet implemented')
def test_reduce_common_subtrees():
    dd = AffTree.identity(2)
    f = AffFunc.from_mats(np.array([[2, 1], [1, 1]], dtype=np.float), np.array([-1, 0], dtype=np.float))
    g = AffFunc.from_mats(np.array([[1, 0], [0, 0]], dtype=np.float), np.array([0, 0], dtype=np.float))

    dd.apply_func(f)
    dd.compose(schema.ReLU(2))
    dd.apply_func(g)
    dd.compose(schema.ReLU(2))
    dd.reduce()

    assert dd.size() == 4


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
    f = AffFunc.from_mats(np.array([[2, 1], [1, -2], [0, 1]], dtype=np.float), np.array([-1, 0, 5], dtype=np.float))
    g = AffFunc.from_mats(np.array([[1, 0, 2], [1, 3, 1], [1, 0, 0]], dtype=np.float), np.array([2, 0, -1], dtype=np.float))

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


@pytest.mark.skip(reason='0 volume elimination is not implemented yet')
def test_infeasible_elimination_trivial_predicates():
    """ Predicates are used multiple times on a path.
    For correct result 0 volume polyhedra must be eliminated. """
    add = AffTree.identity(2)
    f = AffFunc.from_mats(np.array([[2, 1], [1, 1]], dtype=np.float), np.array([-1, 0], dtype=np.float))
    g = AffFunc.from_mats(np.array([[1, 0], [1, 3]], dtype=np.float), np.array([2, 0], dtype=np.float))
    h = AffFunc.from_mats(np.array([[2, 3], [-2, 3], [1, 0]], dtype=np.float),
                       np.array([2, 0, 1], dtype=np.float))

    add.apply_func(f)
    add.compose(schema.ReLU(2))
    add.apply_func(g)
    add.compose(schema.ReLU(2))
    add.apply_func(h)
    add.infeasible_elimination()

    assert add.size() == 9


def test_argmax():
    dd = AffTree.identity(1)
    dd.apply_func(AffFunc.from_mats(np.array([[2], [4], [8]]), np.array([0, -2, -4])))

    dd.compose(schema.argmax(3))

    res = dd.evaluate(np.array([6], dtype=np.float))
    assert res[0] == 2
    
    res = dd.evaluate(np.array([-2], dtype=np.float))
    assert res[0] == 0

