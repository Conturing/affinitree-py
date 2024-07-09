# Welcome to Affinitree!

[![PyPi](https://img.shields.io/pypi/v/affinitree)](https://pypi.org/project/affinitree/)

`Affinitree` is an open-source library designed to generate faithful surrogate models (decision trees) from pre-trained piece-wise linear neural networks.
The core part of the library is implemented in Rust for best performance.

The opaque nature of neural networks stands in the way of their widespread usage, especially in
safety critical domains.
To address this issue, it is important to access the semantics of neural networks in a structured manner.
One promising field of Explainable AI (XAI) research is concerned with finding alternative models for a given neural network
that are more transparent, interpretable, or easier to analyze. 
This approach is called *model distillation* and the resulting model is called *surrogate model*, *proxy model* or *white-box model*.

`Affinitree` enables the distillation of piece-wise linear neural networks into a specialized decision trees.
In Explainable AI (XAI) literature, decision trees are widely regarded as one of the few model classes comprehensible by humans, making them prime candidates for surrogate models.
Additionally, many neural network architectures, such as those using ReLU, LeakyReLU, residual connections, pooling, and convolutions, exhibit piece-wise linear behavior.
These qualities make decision trees well-suited for representing such networks.

Commonly, surrogate models are only an approximation of the true nature of the neural network.
In contrast, `affinitree` provides mathematically sound and correct surrogate models.
This is achieved by an holistic symbolic execution of the network.
The resulting decision structure is human-understandable, but size must be controlled.

# Installation

**affinitree** requires Python 3.8 - 3.12.

```sh
pip install affinitree
```

Wheels are currently available for Linux (x86_64).
For other architectures, see [Build Wheels from Rust](#build-wheels-from-rust).

# First Steps

`Affinitree` provides a high-level API to convert a pre-trained ``pytorch`` model into a decision tree (requires installation of `torch`). 

```python 
from torch import nn
from affinitree import extract_pytorch_architecture, builder

model = nn.Sequential(nn.Linear(7, 5),
                      nn.ReLU(),
                      nn.Linear(5, 5),
                      nn.ReLU(),
                      nn.Linear(5, 4)
                     )

# Train your model here ...

arch = extract_pytorch_architecture(7, model)
dd = builder.from_layers(arch)
```

It may be noted that `affinitree` is independent of any specific neural network library.
The function `extract_pytorch_architecture` is a helper that extracts `numpy` arrays from pytorch models for convenience.
Any model expressed as a sequence of `numpy` matrices can be read (see also the provided [examples](examples)).

After distilling the model, one can use the resulting `AffTree` to plot the decision tree
in [graphviz's](https://graphviz.org/) DOT language:

```python
dd.to_dot()
```

A simple AffTree may look like this (representing the xor function):

<p align="center">
  <img alt="fig:afftree example (at github)" height="300" src="figures/afftree_example.svg"/>
</p>

`Affinitree` provides a method to plot the decision boundaries of an ``AffTree`` using `matplotlib`

```python
from affinitree import plot_preimage_partition, LedgerDiscrete

# Derive 10 colors from the tab10 color map and position legend at the top 
ledger = LedgerDiscrete(cmap='tab10', num=10, position='top')
# Map the terminals of dd to one of the 10 colors based on their class
ledger.fit(dd)
# Plot for each terminal of dd the polytope that is implied by the path from the root to the respective terminal
plot_preimage_partition(dd, ledger, intervals=[(-20., 15.), (-12., 12.)])
```
The ``ledger`` is used to control the coloring and legend of the plot.
A resulting plot may look like this:

<p align="center">
  <img alt="fig:mnist preimage partition (at github)" height="400" src="figures/mnist_preimage_partition.svg"/>
</p>

## Composition and Schemas

Many operations on ``AffTree``s can be implemented using composition.
For example, here are a few common functions in the realm of neural networks expressed as ``AffTree``.

ReLU (indim=4):

<p align="center">
    <img alt="fig:relu" height="400" src="figures/relu_4.svg"/>
</p>

ReLU (indim=4) applied only to the first component (partial ReLU / step ReLU):

<p align="center">
    <img alt="fig:partial-relu" height="150" src="figures/partial_relu_4_0.svg"/>
</p>

Argmax (indim=4):

<p align="center">
    <img alt="fig:argmax" height="300" src="figures/argmax_4.svg"/>
</p>

Schemas are a collection of typical operations that are used in the context of neural networks.
In code, one can apply these as follows:

```python
from affinitree import AffTree
from affinitree import schema

dd = AffTree.identity(2)

relu = schema.ReLU(2)
dd = dd.compose(relu)
```

The following operations are already provided:

```python
schema.ReLU(dim=n)
schema.partial_ReLU(dim=n, row=m)
schema.partial_leaky_ReLU(dim=n, row=m, alpha=0.1)
schema.partial_hard_tanh(dim=n, row=m)
schema.partial_hard_sigmoid(dim=n, row=m)
schema.argmax(dim=n)
schema.inf_norm(minimum=a, maximum=b)
schema.class_characterization(dim=n, clazz=c)
```

The interface is easily adaptable to additional needs, one just needs to define a function that returns an ``AffTree`` instance that expresses the required piece-wise linear function.

# Build Wheels from Rust

For best performance, most of affinitree is written in the system language Rust.
The corresponding sources can be found at [affinitree (rust)](https://github.com/Conturing/affinitree).
To make the interaction with compiled languages easier, Python allows to provide pre-compiled binaries for each target architecture, called *wheels*.

After installing Rust and maturin, wheels for your current architecture can be build using:
```sh
maturin build --release
```

To build wheels optimized for your current system, include the following flag:
```sh
RUSTFLAGS="-C target-cpu=native" maturin build --release
```

An example for setting up a manylinux2014 compatible build environment can be found in the included [Dockerfile](Dockerfile).

# License

Copyright 2022–2024 affinitree developers.

The code in this repository is licensed under the [Apache License, Version 2.0](LICENSE_APACHE). You may not use this project except in compliance with those terms.

Binary applications built from this repository (including wheels) contain dependencies with different license terms, see [license](license.html).

## Contributing

Please feel free to create issues, fork the project, or submit pull requests.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.