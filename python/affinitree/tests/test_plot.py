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

from affinitree import AffTree, AffFunc, Polytope, compute_polytope_vertices
from affinitree import builder

parameter_file = Path(Path(__file__).parent, 'res', 'iris_4-5-5-3.npz')


def test_compute_vertices():
    assert parameter_file.exists()
    
    z = np.zeros((4, 2))
    z[0, 0] = 1.0
    z[2, 1] = 1.0
    func = AffTree.from_aff(AffFunc.from_mats(z, np.zeros(4)))
    poly = AffTree.from_poly(Polytope.hyperrectangle(4, [(-1.0, 1.0)] * 4), AffFunc.identity(4))
    
    func.compose(poly)

    dd = builder.read_npz(2, str(parameter_file), func)
    
    for depth, node, poly in dd.polyhedra():
        if node.is_decision():
            continue
        
        # should not raise any exceptions
        vert = compute_polytope_vertices(poly)
    
    
    
    