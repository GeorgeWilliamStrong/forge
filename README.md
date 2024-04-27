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

## Usage

Please run throught the FWI [demo](https://github.com/GeorgeWilliamStrong/forge/blob/main/tutorials/forge-demo.ipynb) <a target="_blank" href="https://colab.research.google.com/github/GeorgeWilliamStrong/forge/blob/main/tutorials/forge-demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>.

Then run through the breast imaging FWI [example](https://github.com/GeorgeWilliamStrong/forge/tree/main/examples/breast2D) (forward <a target="_blank" href="https://colab.research.google.com/github/GeorgeWilliamStrong/forge/blob/main/examples/breast2D/forward.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a></a>, inverse <a target="_blank" href="https://colab.research.google.com/github/GeorgeWilliamStrong/forge/blob/main/examples/breast2D/inverse.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a></a>).

***
Currently, Forge only supports two-dimensional modelling as it was designed for rapid experimental prototyping. Extending the codes to three-dimensional modelling is trivial in principle, although multi-GPU support has not yet been implemented.
