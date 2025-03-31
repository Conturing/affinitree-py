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

from typing import List, Optional
from affinitree import AffTree, AffFunc, Architecture


def from_layers(layers: Architecture, precondition: Optional['AffTree'], csv: Optional[str]) -> 'AffTree':
    """
    Distills a neural network represented by an Architecture object into an AffTree.

    Parameters
    ----------
    layers : Architecture
        An object containing the layers of the neural network.
    precondition : AffTree, optional (default None)
        An optional precondition to be applied during distillation.
    csv : str, optional (default None)
        A path to a CSV file where the distillation process will be logged.

    Returns
    -------
    AffTree
        The distilled decision tree representing the network.
    """

def read_npz(dim: int, filename: str, precondition: Optional['AffTree'], csv: Optional[str]) -> 'AffTree':
    """
    Reads a `.npz` file containing information about linear layers and activation functions used,
    and distills the encoded network into an AffTree.

    Parameters
    ----------
    dim : int
        The dimensionality of the input to the network.
    filename : str
        The path to the .npz file containing the network information.
    precondition : AffTree, optional (default None)
        An optional precondition to be applied during distillation.
    csv : str, optional (default None)
        A path to a CSV file where the distillation process will be logged.

    Returns
    -------
    AffTree
        The distilled decision tree representing the network.
    """