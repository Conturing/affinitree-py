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

from pathlib import Path
import warnings
from affinitree import AffFunc, Architecture

import numpy as np
try: 
    import torch
except ImportError:
    warnings.warn('PyTorch not available')
    _has_torch = False
else:
    _has_torch = True


def extract_pytorch_architecture(dim: int, model: 'torch.nn.Sequential') -> Architecture:
    if not _has_torch:
        raise ImportError('PyTorch is required')
    arch = Architecture(dim)
    
    layers = model.named_modules(remove_duplicate=False)
    next(layers)

    for _, layer in layers:
        if isinstance(layer, torch.nn.Linear):
            W = layer.weight.detach().numpy().astype(np.float64)
            b = layer.bias.detach().numpy().astype(np.float64)
            arch.linear(AffFunc.from_mats(W, b))
        elif isinstance(layer, torch.nn.ReLU):
            arch.relu()
        elif isinstance(layer, torch.nn.LeakyReLU):
            arch.leaky_relu()
        elif isinstance(layer, torch.nn.Hardsigmoid):
            arch.hard_sigmoid()
        elif isinstance(layer, torch.nn.Hardtanh):
            arch.hard_tanh()
        else:
            print('warning: module \'{layer}\' is currently not supported')
    
    return arch


def export_npz(model: 'torch.nn.Module', filename):
    if not _has_torch:
        raise ImportError('PyTorch is required')
    data = {}
    iter = model.named_modules(remove_duplicate=False)
    next(iter)
    modules = list(iter)
    data['000.layers'] = np.array([len(modules)])
    for idx, (_, layer) in enumerate(modules):
        if isinstance(layer, torch.nn.Linear):
            data[f'{idx:03d}.linear.weights'] = layer.weight.detach().numpy().astype(np.float64)
            data[f'{idx:03d}.linear.bias'] = layer.bias.detach().numpy().astype(np.float64)
        elif isinstance(layer, torch.nn.ReLU):
            data[f'{idx:03d}.relu'] = np.zeros(1)
        elif isinstance(layer, torch.nn.LeakyReLU):
            data[f'{idx:03d}.leakyrelu'] = np.zeros(1)
        elif isinstance(layer, torch.nn.Hardsigmoid):
            data[f'{idx:03d}.hardsigmoid'] = np.zeros(1)
        elif isinstance(layer, torch.nn.Hardtanh):
            data[f'{idx:03d}.hardtanh'] = np.zeros(1)
    path = Path(filename).with_suffix('.npz')
    np.savez(path, **data)