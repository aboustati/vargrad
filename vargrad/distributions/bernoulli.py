import jax.numpy as np

from jax import random


def sample(p, key, num_samples=1):
    """
    Generate Bernoulli Samples
    :param p: Bernoulli probabilities (jax.numpy array)
    :param key: PRNG key
    :param num_samples:  number of samples
    """
    return random.bernoulli(key, p, shape=(num_samples, *p.shape))


def logpmf(x, p):
    """
    Bernoulli log probability mass function
    :param x: outcome (jax.numpy array)
    :param p: Bernoulli probabilities (jax.numpy array)
    """
    assert x.shape == p.shape
    tol = 1e-7
    p = np.clip(p, tol, 1 - tol)
    return x * np.log(p + tol) + (1 - x) * np.log(1 - p + tol)
