import jax.numpy as np

from jax import random, nn
from jax.scipy.special import logit


def sample(p, key, num_samples=1):
    """
    Generate Binomial Concrete samples
    :param p: Binary relaxed params (interpreted as Bernoulli probabilities) (jax.numpy array)
    :param key: PRNG key
    :param num_samples: number of samples
    """
    tol = 1e-7
    p = np.clip(p, tol, 1 - tol)
    logit_p = logit(p)
    u = random.uniform(key, shape=(num_samples, *p.shape))
    logit_u = logit(np.clip(u, tol, 1-tol))
    return logit_p + logit_u


def conditional_sample(p, y, key):
    """
    Generate conditional Binary relaxed samples
    :param p: Binary relaxed params (interpreted as Bernoulli probabilities) (jax.numpy array)
    :param y: Conditioning parameters (jax.numpy array)
    :param key: PRNG key
    """
    tol = 1e-7
    p = np.clip(p, tol, 1 - tol)

    v = random.uniform(key, shape=y.shape)
    v_prime = (v * p + (1 - p)) * y + (v * (1 - p)) * (1 - y)
    v_prime = np.clip(v_prime, tol, 1 - tol)

    logit_v = logit(v_prime)
    logit_p = logit(p)
    return logit_p + logit_v
