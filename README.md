# Forge

![Build Status](https://github.com/GeorgeWilliamStrong/forge/actions/workflows/tutorials.yml/badge.svg)

A full-waveform inversion (FWI) framework built in [PyTorch](https://pytorch.org/). Forge was designed and built for research and prototyping of tomographic acoustic imaging methods.

Everything has been built from the ground up using pure PyTorch, with no third-party libraries relied upon for wave propagation or inversion. This gives the user complete control and customisability.

**Forge was built to:**
1. accelerate FWI research,
2. allow seamless integration with cutting-edge machine learning techniques and models,
4. leverage GPU acceleration,
5. quickly forge prototypes.

**Design philosophy:**
1. be easy to read, understand and use,
2. be *entirely* hackable,
3. built on PyTorch.

If three-dimensional modelling and/or production grade performance are required, [stride](https://www.stride.codes/) and [devito](https://www.devitoproject.org/) are recommended.

## Quickstart

Install Forge and its required dependencies as follows:
```sh
git clone https://github.com/GeorgeWilliamStrong/forge.git
cd forge
pip install -e .
```

Plotting functionality used in the demos employs [PySlice](https://github.com/GeorgeWilliamStrong/pyslice), which can be installed as follows:

```sh
git clone https://github.com/GeorgeWilliamStrong/pyslice
cd pyslice
pip install -e .
```

### Examples

Please refer to the FWI [demo](https://github.com/GeorgeWilliamStrong/forge/blob/main/tutorials/forge-demo.ipynb) <a target="_blank" href="https://colab.research.google.com/github/GeorgeWilliamStrong/forge/blob/main/tutorials/forge-demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> for a detailed overview of how to use Forge.

There is also a breast imaging FWI example located [here](https://github.com/GeorgeWilliamStrong/forge/tree/main/examples/breast2D) (forward <a target="_blank" href="https://colab.research.google.com/github/GeorgeWilliamStrong/forge/blob/main/examples/breast2D/forward.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a></a>, inverse <a target="_blank" href="https://colab.research.google.com/github/GeorgeWilliamStrong/forge/blob/main/examples/breast2D/inverse.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a></a>).

### Usage

```python
from forge.model import FullWaveformInversion
import torch

# Instantiate model using the starting model
model = FullWaveformInversion(model = m0,
                              dx = dx,
                              dt = dt,
                              r_pos = r_pos)

# Instantiate PyTorch optimizer
opt = torch.optim.SGD([model.m], lr=1e-5, momentum=0.4)

# Define loss function
l2_loss = torch.nn.MSELoss()

# Run the optimisation loop
model.fit(data = true_model.d,
          s_pos = s_pos,
          source = source,
          optimizer = opt,
          loss = l2_loss,
          num_iter = 10,
          bs = 10,
          blocks = [1e5, 2e5])
```

***
Currently, Forge only supports two-dimensional modelling as it was designed for rapid experimental prototyping. Extending the codes to three-dimensional modelling is trivial in principle, although multi-GPU support has not yet been implemented.
