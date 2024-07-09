from affinitree import *

import numpy as np

f = AffFunc.from_mats(np.array([[1, 0], [1, 3]]), np.array([2, 0]))
g = AffFunc.from_mats(np.array([[2, -1], [3, 1]]), np.array([1, -2]))
h = AffFunc.from_mats(np.array([[2, 3], [-2, 3], [1, 0]]), np.array([2, 0, 1]))

dd = AffTree.identity(2)
dd.apply_func(f)
dd.compose(schema.ReLU(2))
dd.apply_func(g)
dd.compose(schema.ReLU(2))
dd.apply_func(h)
dd.compose(schema.class_characterization(3, 0))

dd.infeasible_elimination()

# print dot string to console
print(dd.to_dot())
