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

from affinitree import schema


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
