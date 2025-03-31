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
import numpy as np
import torch
from torch import nn

from affinitree import AffTree, AffFunc, Polytope, extract_pytorch_architecture
from affinitree import builder

parameter_file = Path(Path(__file__).parent, 'res', 'iris_4-5-5-3.npz')

def assert_equiv_net(model, dd):
    torch.manual_seed(42)
    rnd = np.random.default_rng(42)

    for idx in range(10000):
        x = 10 * rnd.random(4, dtype=np.float32)

        net_out = model.forward(torch.from_numpy(x))
        dd_out = dd.evaluate(x)

        assert torch.allclose(net_out, torch.from_numpy(dd_out).to(torch.float32), atol=1e-05)


def test_read_npz():
    assert parameter_file.exists()
    
    dnn = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 3))
    dnn.load_state_dict(torch.load(parameter_file.with_suffix('.pt')))
    
    dd = builder.read_npz(4, str(parameter_file))
    
    assert_equiv_net(dnn, dd)


def test_read_npz_pre():
    assert parameter_file.exists()
    
    dnn = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 3))
    dnn.load_state_dict(torch.load(parameter_file.with_suffix('.pt')))
    
    pre = AffTree.from_poly(Polytope.hyperrectangle(4, [(-1., 11.)] * 4), AffFunc.identity(4))
    dd = builder.read_npz(4, str(parameter_file), pre)
    
    assert_equiv_net(dnn, dd)


def test_from_layers():
    assert parameter_file.exists()
    
    dnn = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 3))
    dnn.load_state_dict(torch.load(parameter_file.with_suffix('.pt')))
    
    arch = extract_pytorch_architecture(4, dnn)
    dd = builder.from_layers(arch)
    
    assert_equiv_net(dnn, dd)


def test_from_layers_pre():
    assert parameter_file.exists()
    
    dnn = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 3))
    dnn.load_state_dict(torch.load(parameter_file.with_suffix('.pt')))
    
    pre = AffTree.from_poly(Polytope.hyperrectangle(4, [(-1., 11.)] * 4), AffFunc.identity(4))
    arch = extract_pytorch_architecture(4, dnn)
    dd = builder.from_layers(arch, pre)
    
    assert_equiv_net(dnn, dd)