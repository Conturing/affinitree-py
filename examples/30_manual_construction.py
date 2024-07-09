import numpy as np
from affinitree import AffTree, AffFunc
from affinitree import schema

w1 = np.array([[1, 1], [2, 2]])
b1 = np.array([3, 3])
l1 = AffFunc.from_mats(w1, b1)

w2 = np.array([[1, 2], [2, 1]])
b2 = np.array([0, 0])
l2 = AffFunc.from_mats(w2, b2)

# construct AffTree from AffFunc
dd = AffTree.identity(2)
dd.apply_func(l1)
dd.compose(schema.ReLU(2))
dd.apply_func(l2)

# print dot string to console
print(dd.to_dot())
