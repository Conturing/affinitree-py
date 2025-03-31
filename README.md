[![PyPi](https://img.shields.io/pypi/v/affinitree)](https://pypi.org/project/affinitree/)
[![Build](https://img.shields.io/github/actions/workflow/status/Conturing/affinitree/ci.yml)](https://github.com/Conturing/affinitree/actions)
[![License](https://img.shields.io/github/license/Conturing/affinitree)](LICENSE-APACHE)

# Affinitree

`Affinitree` is an open-source library designed to generate *interpretable decision trees* from *pre-trained, piecewise linear neural networks* (like ReLU-based DNNs).
These decision trees act as *faithful surrogate models*, offering an interpretable yet functionally-equivalent alternative to the typically opaque nature of deep neural networks.
The core part of the library is implemented in [Rust](https://github.com/Conturing/affinitree) for best performance.

# Features

Currently the following features are supported:
 - Construction of functionally equivalent decision trees from deep neural networks with piecewise linear activation functions (e.g., ReLU, Leaky ReLU, Hard Tanh, Hard Sigmoid)
 - Modular and extensible API making use of the compositionality of linear algebra
 - On-the-fly redundancy elimination of infeasible paths
 - Visualizations using Graphviz's DOT language and matplotlib

# Installation

**Affinitree** requires Python 3.8 - 3.12.

```sh
pip install affinitree
```

Pre-built wheels are currently available for Linux, Windows, and MacOS.
For other architectures, see the instructions below on how to [build wheels from Rust](#build-wheels-from-rust).

# First Steps

`Affinitree` provides a high-level API to convert pre-trained `PyTorch` models into decision trees.
The example below demonstrates how to extract a PyTorch model's architecture and build (*distill*) an interpretable decision tree from it:

```python 
from torch import nn
from affinitree import extract_pytorch_architecture, builder

in_dim = 7
model = nn.Sequential(nn.Linear(in_dim, 5),
                      nn.ReLU(),
                      nn.Linear(5, 5),
                      nn.ReLU(),
                      nn.Linear(5, 4)
                     )

# Train your model here ...

arch = extract_pytorch_architecture(in_dim, model)
tree = builder.from_layers(arch)
```

It may be noted that `affinitree` is independent of any specific neural network library.
The function `extract_pytorch_architecture` is a helper that extracts `numpy` arrays from `PyTorch` models for convenience.
Any model expressible as a sequence of `numpy` matrices can be read (see also the provided [examples](examples)).

## Visualizing the Decision Tree
After distilling the model, you can use the resulting `AffTree` to plot the decision tree
in [graphviz's](https://graphviz.org/) DOT format:

```python
tree.to_dot()
```

The emitted DOT code can be rendered into an image such as:

<p align="center">
  <img alt="fig:afftree example (at GitHub)" height="300" src="figures/afftree_example.svg"/>
</p>

## Plotting Decision Boundaries
`Affinitree` provides methods to plot the decision boundaries of an `AffTree` using `matplotlib`.
In a classifier, decision boundaries are boundaries in the input space that separate regions where the prediction of the classifier changes.
Therefore they constitute an important tool for understanding its decision making. 

```python
from affinitree import plot_preimage_partition, LedgerDiscrete

tree = # ...

# ...

# Derive 10 colors from the tab10 color map and position legend at the top 
ledger = LedgerDiscrete(cmap='tab10', num=10, position='top')
# Map the terminals of dd to one of the 10 colors based on their class
ledger.fit(tree)
# Plot for each terminal of dd the polytope that is implied by the path from the root to the respective terminal
plot_preimage_partition(tree, ledger, intervals=[(-20., 15.), (-12., 12.)])
```
The `ledger` is used to control the coloring and legend of the plot.
A resulting plot may look like this:

<p align="center">
  <img alt="fig:mnist preimage partition (at GitHub)" height="400" src="figures/mnist_preimage_partition.svg"/>
</p>

## Composition and Schemas

Many complex operations on `AffTree`s can be constructed in a modular fashion using the `composition` function.
In fact, the previously discussed higher-level functions for distillation internally rely on composition.
This is powered by a collection of predefined `AffTree`s that represent commonly used functions in deep learning.
These are implemented in the `schema` module and include currently:
```python
schema.partial_ReLU(dim=n, row=m)
schema.partial_leaky_ReLU(dim=n, row=m, alpha=0.1)
schema.partial_hard_tanh(dim=n, row=m)
schema.partial_hard_sigmoid(dim=n, row=m)
schema.argmax(dim=n)
schema.inf_norm(minimum=a, maximum=b)
schema.class_characterization(dim=n, clazz=c)
```
Here $n$ refers to the input dimension and $row$ to a single neuron of the current layer.

For example, the `AffTree` that represents the partial ReLU function $\mathbb{R}^4 \to \mathbb{R}^4$ that acts only on the first neuron is given by:
<p align="center">
    <img alt="fig:partial-relu" height="150" src="figures/partial_relu_4_0.svg"/>
</p>

In code, the composition looks as follows:
```python
from affinitree import AffTree, schema

tree = AffTree.identity(4)

relu = schema.partial_ReLU(dim=4, row=0)
tree.compose(relu)
```

The interface is easily adaptable to additional needs, one just needs to define a function that returns an `AffTree` instance that expresses the required piecewise linear function.

# Academic Background

Despite their operational effectiveness, deep neural networks (DNNs) are typically characterized as opaque, black-box systems due to their non-linear, sub-symbolic nature, and inherent complexity.
As a consequence, their adherence to desirable semantic properties like safety, fairness, legal-compliance, and control cannot be guaranteed, which slows down their adoption in safety-critical applications.
To address these issues, it is important to gain access to their semantics in a structured manner.
Building on recent work in the field of [Explainable AI (XAI)](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence), `affinitree` implements a technique known as *model distillation* where a separate, more interpretable model is constructed to mimic a given opaque system (DNNs in this case).
In XAI literature, decision trees are widely regarded as one of a few model classes comprehensible for humans, making them a prime candidate for such a surrogate model.
In contrast to other distillation approaches, `affinitree` distills piecewise linear DNNs into *faithful* (that is semantically-equivalent) surrogate models.
Therefore the resulting surrogate models can be analyzed with mathematical precision, allowing to derive hard guarantees for the behavior of the original DNNs.

Technically, `affinitree` enables the distillation of piecewise linear neural networks into specialized decision trees that extend [BSP Trees](https://en.wikipedia.org/wiki/Binary_space_partitioning) with a suite of algebraic properties (mostly oriented at linear algebra).
When the given DNN is piecewise linear these algebraic properties can be defined by decomposing the DNN into its linear pieces.
Today many neural network architectures - such as those using ReLU, LeakyReLU, residual connections, pooling, and convolutions - exhibit piecewise linear behavior.
Based on these algebraic features, `affinitree` provides a modular API that simplifies usage and customization.
`Affinitree` provides several optimization techniques - including infeasible path elimination, redundancy elimination, and concolic execution - to improve the size of the resulting surrogate models and with that the running time of distillation.

# Build Wheels from Rust

For best performance, most of `affinitree` is written in the system language Rust.
The corresponding sources can be found at [affinitree (rust)](https://github.com/Conturing/affinitree).
To make the interaction with compiled languages easier, Python allows to provide pre-compiled binaries for each target architecture, called *wheels*.

After installing Rust and [`maturin`](https://github.com/PyO3/maturin), wheels for your current architecture can be built using:
```sh
maturin build --release
```

To build wheels optimized for your current system, include the following flag:
```sh
RUSTFLAGS="-C target-cpu=native" maturin build --release
```

An example for setting up a manylinux2014 compatible build environment can be found in the included [Dockerfile](Dockerfile).

# License

Copyright 2022–2025 affinitree developers.

The code in this repository is licensed under the [Apache License, Version 2.0](LICENSE-APACHE). You may not use this project except in compliance with those terms.

Binary applications built from this repository, including wheels, contain dependencies with different license terms, see [binary license](binary-license.html).

## Contributing

Please feel free to create issues, fork the project, or submit pull requests!

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the [Apache-2.0](LICENSE-APACHE) license, shall be licensed as above, without any additional terms or conditions.