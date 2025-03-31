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

from typing import Optional
from .. import AffTree

def partial_ReLU(dim: int, row: int) -> AffTree:
    """
    Creates an AffTree instance representing the ReLU function applied to a specific `row`.

    Parameters
    ----------
    dim : int
        The dimension of the input space.
    row : int
        The row on which the ReLU is applied.

    Notes
    -----
    .. math:

        \text{partial\_ReLU}(x)_i = \begin{cases}
        \max(0, x_i) & \text{if } i = \text{row} \\
        x_i & \text{otherwise}
        \end{cases}
    """

def partial_leaky_ReLU(dim: int, row: int, alpha: float) -> AffTree:
    """
    Creates an AffTree instance representing the Leaky ReLU function applied to a specific `row`.

    Parameters
    ----------
    dim : int
        The dimension of the input space.
    row : int
        The row on which the partial leaky ReLU is applied.
    alpha : float
        The slope of the function for negative input values.

    Notes
    -----
    .. math::

        \text{partial\_leaky\_ReLU}(x)_i = \begin{cases}
        x_i & \text{if } i = \text{row} \land x_i \geq 0 \\
        \alpha x_i & \text{if } i = \text{row} \land x_i < 0 \\
        x_i & \text{otherwise}
        \end{cases}
    """

def partial_hard_tanh(dim: int, row: int, min_val: float, max_val: float) -> AffTree:
    """
    Creates an AffTree instance representing the hard tanh function applied to a specific `row`.

    Parameters
    ----------
    dim : int
        The dimension of the input space.
    row : int
        The row on which the partial hard tanh is applied.
    min_val : float
        The minimum value of the hard tanh function.
    max_val : float
        The maximum value of the hard tanh function.

    Notes
    -----
    .. math::

        \text{partial\_hard\_tanh}(x)_i = \begin{cases}
        \min(\max(x_i, \text{min\_val}), \text{max\_val}) & \text{if } i = \text{row} \\
        x_i & \text{otherwise}
        \end{cases}
    """

def partial_hard_sigmoid(dim: int, row: int) -> AffTree:
    """
    Creates an AffTree instance representing the hard sigmoid function applied to a specific `row`.

    Parameters
    ----------
    dim : int
        The dimension of the input space.
    row : int
        The row on which the partial hard sigmoid is applied.

    Notes
    -----
    .. math::

        \text{partial\_hard\_sigmoid}(x)_i = \begin{cases}
        \max(0, \min(1, \frac{x_i + 1}{2})) & \text{if } i = \text{row} \\
        x_i & \text{otherwise}
        \end{cases}
    """

def argmax(dim: int) -> AffTree:
    """
    Creates an AffTree instance representing the argmax function.

    Parameters
    ----------
    dim : int
        The dimension of the input space.

    Notes
    -----
    .. math::

        \text{argmax}(x) = \text{index of the maximum value in } x
    """

def class_characterization(dim: int, clazz: int) -> AffTree:
    """
    Creates an AffTree instance representing the class characterization function.

    Parameters
    ----------
    dim : int
        The dimension of the input space.
    clazz : int
        The class index to be characterized.

    Notes
    -----
    .. math::

        \text{class\_characterization}(x, \text{clazz}) = \begin{cases}
        1 & \text{if } \forall j \neq \text{clazz} : x_\text{clazz} \geq x_j \\
        0 & \text{otherwise}
        \end{cases}
    """

def inf_norm(dim: int, minimum: Optional[float] = None, maximum: Optional[float] = None) -> AffTree:
    """
    Creates an AffTree instance representing the infinity norm function.

    Parameters
    ----------
    dim : int
        The dimension of the input space.
    minimum : float, optional
        The minimum value constraint.
    maximum : float, optional
        The maximum value constraint.

    Notes
    -----
    .. math::

        \text{inf\_norm}(x) = \max(|x|)
    """
