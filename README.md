# Forge

A full-waveform inversion framework built in [PyTorch](https://pytorch.org/).

**Forge was built to:**
1. accelerate research,
2. allow seamless integration with cutting-edge machine learning techniques,
4. leverage GPU acceleration,
5. forge prototypes quickly.

**Design philosophy:**
1. be easy to read, understand and use,
2. be *entirely* hackable,
3. built on PyTorch.

Forge is designed and built for research and prototyping. If three-dimensional modelling and/or production grade performance are required, use [stride](https://www.stride.codes/).

## Quickstart

```sh
git clone https://github.com/GeorgeWilliamStrong/forge.git
cd forge
pip install -e .
```

Then run through the [demo](https://github.com/GeorgeWilliamStrong/forge/blob/main/examples/forge-demo.ipynb). 

### What Forge isn't for
Currently, Forge only supports two-dimensional modelling. Extending the code to three-dimensional modelling is trivial. Scalability of three-dimensional modelling is **not trivial**. However, with the recent rapid rise in easy-to-use distributed PyTorch training frameworks (e.g. [Ray](https://docs.ray.io/en/latest/index.html), [Horovod](https://horovod.ai/) and [PyTorch Lightning](https://www.pytorchlightning.ai/)), perhaps scalability is not as far off as it may seem...
