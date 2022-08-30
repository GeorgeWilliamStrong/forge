# Forge

A full-waveform inversion framework built in [PyTorch](https://pytorch.org/), designed to: 

1. accelerate research,
2. allow seamless integration with cutting-edge machine learning techniques,
3. be *easily* and *entirely* hackable,
4. leverage GPU acceleration,
5. forge prototypes quickly.

#

Currently, Forge only supports two-dimensional modelling. Extending the code to three-dimensional modelling is trivial. Scalability of three-dimensional modelling is not trivial, yet, with the recent rapid rise in easy-to-use distributed PyTorch training frameworks (e.g. [Ray](https://docs.ray.io/en/latest/index.html), [Horovod](https://horovod.ai/) and [PyTorch Lightning](https://www.pytorchlightning.ai/)), perhaps scalability is not as far off as it may seem.
