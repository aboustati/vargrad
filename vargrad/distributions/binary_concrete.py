import jax.numpy as np

from jax import random, nn
from jax.scipy.special import logit


def sample(p, temperature, key, num_samples=1):
    """
    Generate Binomial Concrete samples
    :param p: Binomial Concrete params (interpreted as Bernoulli probabilities) (jax.numpy array)
    :param temperature: temperature parameter
    :param key: PRNG key
    :param num_samples: number of samples
    """
    tol = 1e-7
    p = np.clip(p, tol, 1 - tol)
    logit_p = logit(p)
    base_randomness = random.logistic(key, shape=(num_samples, *p.shape))
    return nn.sigmoid((logit_p + base_randomness) / (temperature + tol))


def logpdf(x, p, temperature):
    """
    Bernoulli log probability mass function
    :param x: outcome (jax.numpy array)
    :param p: Binomial Concrete params (interpreted as Bernoulli probabilities) (jax.numpy array)
    :param temperature: temperature parameter
    """
    assert x.shape == p.shape
    tol = 1e-7
    p = np.clip(p, tol, 1 - tol)
    x = np.clip(x, tol, 1 - tol)
    logit_p = logit(p)
    first_term = np.log(temperature) + logit_p - (1 + temperature) * np.log(x) - (1 + temperature) * np.log(1 - x)
    second_term = 2 * np.log((np.exp(logit_p) * (x ** (- temperature))) + (1 - x) ** (- temperature))
    return first_term - second_term


def conditional_sample(p, y, temperature, key):
    """
    Generate conditional Binomial Concrete sample
    :param p: Binomial Concrete params (interpreted as Bernoulli probabilities) (jax.numpy array)
    :param y: Conditioning parameters (jax.numpy array)
    :param temperature: temperature parameter
    :param key: PRNG key
    """
    tol = 1e-7
    p = np.clip(p, tol, 1 - tol)

    v = random.uniform(key, shape=y.shape)
    v_prime = (v * p + (1 - p)) * y + (v * (1 - p)) * (1 - y)
    v_prime = np.clip(v_prime, tol, 1 - tol)

    logit_v = logit(v_prime)
    logit_p = logit(p)
    return nn.sigmoid((logit_p + logit_v) / (temperature + tol))
