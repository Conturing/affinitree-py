import affinitree as aff
from affinitree.plot import plot_preimage_partition, plot_image, LedgerContinuous

import numpy as np
import matplotlib.pyplot as plt

dd = aff.AffTree.from_array(np.array([[1., 2.], [2., 1.]]), np.array([0.5, -0.5]))

dd.compose(aff.schema.ReLU(2))
dd.apply_func(aff.AffFunc.from_mats(np.array([[2., 1.], [-0.5, 1.]]), np.array([1., -2.])))
dd.compose(aff.schema.ReLU(2))

print(dd)

ledger = LedgerContinuous()
ledger.fit(dd)
plot_preimage_partition(dd, ledger, intervals=[(-20., 20.), (-20., 20.)])
plt.savefig('relu-regions.png')

plot_image(dd, ledger, intervals=[(-20., 20.), (-20., 20.), (-5, 60)], projection_out=lambda x: x[:,0])
plt.savefig('relu-image.png')


