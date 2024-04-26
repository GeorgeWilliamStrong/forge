# Forge

A full-waveform inversion (FWI) framework built in [PyTorch](https://pytorch.org/). Forge was designed and built for research and prototyping of tomographic acoustic imaging methods.

Everything has been built from the ground up in pure PyTorch, with no third-party libraries relied upon for wave propagation or inversion. This gives the user complete control and customisability.

**Forge was built to:**
1. accelerate FWI research,
2. allow seamless integration with cutting-edge machine learning techniques and models,
4. leverage GPU acceleration,
5. quickly forge prototypes.

**Design philosophy:**
1. be easy to read, understand and use,
2. be *entirely* hackable,
3. built on PyTorch.

If three-dimensional modelling and/or production grade performance are required, [stride](https://www.stride.codes/) is recommended.

## Quickstart

```sh
git clone https://github.com/GeorgeWilliamStrong/forge.git
cd forge
pip install -e .
```

Plotting functionality used in the demos employs PySlice, which can be installed as follows:

```sh
conda activate forge
git clone https://github.com/GeorgeWilliamStrong/pyslice
cd pyslice
pip install -e .
```

Then run through the [demo](https://github.com/GeorgeWilliamStrong/forge/blob/main/examples/forge-demo.ipynb).

## Theory



### What Forge isn't for
Currently, Forge only supports two-dimensional modelling as it was designed for experimental prototyping. Extending the codes to three-dimensional modelling is trivial in principle, although multi-GPU support has not yet been implemented.
