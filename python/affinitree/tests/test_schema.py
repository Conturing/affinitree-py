import numpy as np

from affinitree import schema


def test_relu():
    relu_dd = schema.ReLU(4)

    assert np.allclose(relu_dd.evaluate(np.array([-1, -100, -2, 1000])), np.array([0, 0, 0, 1000]))
    assert np.allclose(relu_dd.evaluate(np.array([1, 2, -2, 1])), np.array([1, 2, 0, 1]))
    assert np.allclose(relu_dd.evaluate(np.array([1, 0, 0, 4])), np.array([1, 0, 0, 4]))


def test_partial_relu():
    relu_dd = schema.partial_ReLU(4, 2)

    assert np.allclose(relu_dd.evaluate(np.array([-1, -100, -2, 1000])), np.array([-1, -100, 0, 1000]))
    assert np.allclose(relu_dd.evaluate(np.array([1, 2, -2, 1])), np.array([1, 2, 0, 1]))
    assert np.allclose(relu_dd.evaluate(np.array([1, 0, 0, 4])), np.array([1, 0, 0, 4]))

    relu_dd = schema.partial_ReLU(4, 1)

    assert np.allclose(relu_dd.evaluate(np.array([-1, -100, -2, 1000])), np.array([-1, 0, -2, 1000]))
    assert np.allclose(relu_dd.evaluate(np.array([1, 2, -2, 1])), np.array([1, 2, -2, 1]))
    assert np.allclose(relu_dd.evaluate(np.array([1, 0, 0, 4])), np.array([1, 0, 0, 4]))


def test_argmax():
    relu_dd = schema.argmax(4)

    assert relu_dd.evaluate(np.array([1, 2, -2, 1])).item() == 1
    assert relu_dd.evaluate(np.array([1, 0, 0, 4])).item() == 3


def test_class_characterization():
    relu_dd = schema.class_characterization(4, 3)

    assert relu_dd.evaluate(np.array([1, 2, -2, 1])).item() == 0
    assert relu_dd.evaluate(np.array([1, 0, 0, 4])).item() == 1

    relu_dd = schema.class_characterization(4, 1)

    assert relu_dd.evaluate(np.array([1, 2, -2, 1])).item() == 1
    assert relu_dd.evaluate(np.array([1, 0, 0, 4])).item() == 0


def test_inf_norm():
    relu_dd = schema.inf_norm(4, maximum=3, minimum=-2)

    assert relu_dd.evaluate(np.array([1, 2, -2, 1])).item() == 1
    assert relu_dd.evaluate(np.array([1, 0, 0, 4])).item() == 0
