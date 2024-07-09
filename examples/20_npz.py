from pathlib import Path
from affinitree import AffTree, AffFunc
from affinitree import builder
from affinitree.pytorch import export_npz

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 52
rnd = torch.random.manual_seed(seed)
np.random.seed(seed)
print(f'Using seed {seed}')

# neural network in pytorch
model = nn.Sequential(
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 3)
)

# optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# load dataset from scipy
iris = load_iris()
train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.20)

# normalize for best pratice
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

train_td = TensorDataset(torch.from_numpy(train_x).type(torch.float32), torch.from_numpy(train_y).type(torch.long))
test_td = TensorDataset(torch.from_numpy(test_x).type(torch.float32), torch.from_numpy(test_y).type(torch.long))

train_dl = DataLoader(train_td, pin_memory=True, batch_size=10, shuffle=True)
test_dl = DataLoader(test_td, pin_memory=True, batch_size=5)

# training loop
for epoch in range(20):
    for batch, labels in train_dl:
        pred = model(batch)
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# export the learned parameters using the numpy 'npz' format
export_npz(model, 'examples/nn-iris-4-4-3.npz')

# read in the parameters and architecture and construct an AffTree 
dd = builder.read_npz(4, 'examples/nn-iris-4-4-3.npz')

print(dd.to_dot())