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

from pathlib import Path
from affinitree import AffFunc, LayerBuilder

import numpy as np
from torch import nn


def extract_pytorch_architecture(dim: int, model: nn.Sequential) -> LayerBuilder:
    arch = LayerBuilder(dim)
    
    layers = model.named_modules(remove_duplicate=False)
    next(layers)

    for _, layer in layers:
        if isinstance(layer, nn.Linear):
            W =  layer.weight.detach().numpy().astype(np.float64)
            b = layer.bias.detach().numpy().astype(np.float64)
            arch.linear(AffFunc.from_mats(W, b))
        elif isinstance(layer, nn.ReLU):
            arch.relu()
    
    return arch


def export_npz(model: nn.Module, filename):
        data = {}
        iter = model.named_modules(remove_duplicate=False)
        next(iter)
        modules = list(iter)
        data['000.layers'] = np.array([len(modules)])
        for idx, (_, layer) in enumerate(modules):
            if isinstance(layer, nn.Linear):
                data[f'{idx:03d}.linear.weights'] = layer.weight.detach().numpy().astype(np.float64)
                data[f'{idx:03d}.linear.bias'] = layer.bias.detach().numpy().astype(np.float64)
            elif isinstance(layer, nn.ReLU):
                data[f'{idx:03d}.relu'] = np.zeros(1)
        path = Path(filename).with_suffix('.npz')
        np.savez(path, **data)