from ..distributions import bernoulli, binary_concrete, binary_relaxed

import jax
import jax.numpy as np

from jax import random, lax, vmap
from jax.tree_util import tree_map, tree_reduce


def _bernoulli_log_likelihood(decoder):
    def bernoulli_log_likelihood(decoder_params, sample, batch):
        y = batch
        y_pred = decoder(decoder_params, sample)  # BxD
        logprob = bernoulli.logpmf(y, y_pred)  # BxD
        return np.sum(logprob)
    return bernoulli_log_likelihood


def generate_negative_elbo(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder)

    def negative_elbo(encoder_params, decoder_params, batch, prng_key, num_samples=1):
        """
        Computes the ELBO of a discrete VAE
        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param num_samples: number of samples
        """
        posterior_params = encoder(encoder_params, batch)  # BxD
        posterior_samples = bernoulli.sample(posterior_params, prng_key, num_samples)

        log_prior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(posterior_samples, 0.5 * np.ones(
                (batch.shape[0], posterior_params.shape[-1])
            )),
            axis=(1, 2)
        )  # SxBxD

        log_posterior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(posterior_samples, posterior_params),
            axis=(1, 2)
        )

        log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
            decoder_params, posterior_samples, batch
        )

        elbo_samples = log_likelihood - log_posterior + log_prior
        return - np.mean(elbo_samples, axis=0) / batch.shape[0]
    return negative_elbo


def generate_score_function_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder)

    def score_function_objective(encoder_params, decoder_params, batch, prng_key, num_samples=1):
        """
        Computes the score function objective of a discrete VAE.
        The gradient of this objective matches the core function gradient of the ELBO.
        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param num_samples: number of samples
        """
        posterior_params = encoder(encoder_params, batch)  # BxD
        posterior_samples = lax.stop_gradient(bernoulli.sample(posterior_params, prng_key, num_samples))

        log_prior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(posterior_samples, 0.5 * np.ones(
                (batch.shape[0], posterior_params.shape[-1])
            )),
            axis=(1, 2)
        )  # SxBxD

        log_posterior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(posterior_samples, posterior_params),
            axis=(1, 2)
        )

        log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
            decoder_params, posterior_samples, batch
        )

        elbo_samples = log_likelihood - log_posterior + log_prior
        elbo_samples = lax.stop_gradient(elbo_samples) * log_posterior
        return - np.mean(elbo_samples, axis=0) / batch.shape[0]
    return score_function_objective


def generate_variance_loss_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder)

    def variance_loss_objective(encoder_params, decoder_params, batch, prng_key, num_samples=1):
        """
        Computes the variance loss objective of a discrete VAE.
        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param num_samples: number of samples
        """
        posterior_params = encoder(encoder_params, batch)  # BxD
        posterior_samples = lax.stop_gradient(bernoulli.sample(posterior_params, prng_key, num_samples))

        log_prior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(posterior_samples, 0.5 * np.ones(
                (batch.shape[0], posterior_params.shape[-1])
            )),
            axis=(1, 2)
        )  # SxBxD

        log_posterior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(posterior_samples, posterior_params),
            axis=(1, 2)
        )

        log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
            decoder_params, posterior_samples, batch
        )
        elbo_samples = log_likelihood - log_posterior + log_prior
        return np.var(elbo_samples, axis=0, ddof=1) / batch.shape[0]
    return variance_loss_objective


def generate_concrete_relaxation_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder)

    def concrete_relaxation_objective(encoder_params, decoder_params, batch, prng_key,
                                      temperatures=(1/2, 2/3), num_samples=1):
        """
        Computes the Concrete relaxation objective function of a discrete VAE.
        This is a standard reparameterisation objective with a relaxed distribution.
        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param temperatures: temperature parameters for the prior and posterior respectively (tuple).
        :param num_samples: number of samples
        """

        posterior_params = encoder(encoder_params, batch)  # BxD
        posterior_samples = binary_concrete.sample(posterior_params, temperatures[1], prng_key, num_samples)

        log_prior = np.sum(
            vmap(lambda x, y: binary_concrete.logpdf(x, y, temperatures[0]), in_axes=(0, None))(
                posterior_samples, 0.5 * np.ones((batch.shape[0], posterior_params.shape[-1]))
            ),
            axis=(1, 2)
        )  # SxBxD

        log_posterior = np.sum(
            vmap(lambda x, y: binary_concrete.logpdf(x, y, temperatures[1]), in_axes=(0, None))(
                posterior_samples, posterior_params
            ),
            axis=(1, 2)
        )

        log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
            decoder_params, posterior_samples, batch
        )
        elbo_samples = log_likelihood - log_posterior + log_prior
        return - np.mean(elbo_samples, axis=0) / batch.shape[0]
    return concrete_relaxation_objective


def generate_rebar_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder)

    def rebar_objective(encoder_params, decoder_params, batch, prng_key, cv_coeff=1.0, temperature=0.5, num_samples=1):
        """
        Computes the REBAR objective function of a discrete VAE.

        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param cv_coeff: control variate coefficient
        :param temperature: temperature parameter for rebar control variate.
        :param num_samples: number of samples
        """
        posterior_params = encoder(encoder_params, batch)  # BxD

        # Sampling
        sampling_key, control_variate_key = random.split(prng_key)
        # posterior_samples = bernoulli.sample(posterior_params, sampling_key, num_samples)
        # concrete_samples = binary_concrete.sample(posterior_params, temperature, sampling_key, num_samples)
        relaxed_samples = binary_relaxed.sample(posterior_params, sampling_key, num_samples)

        posterior_samples = np.heaviside(relaxed_samples, 0)
        concrete_samples = jax.nn.sigmoid(relaxed_samples / temperature)

        cv_samples = binary_concrete.conditional_sample(posterior_params, posterior_samples,
                                                        temperature, control_variate_key)

        log_posterior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(posterior_samples, posterior_params),
            axis=(1, 2)
        )

        def elbo_samples(samples):
            log_prior = np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(samples, 0.5 * np.ones((batch.shape[0],
                                                                                  posterior_params.shape[-1]))),
                axis=(1, 2)
            )  # SxBxD

            log_posterior = np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(samples, posterior_params),
                axis=(1, 2)
            )

            log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
                decoder_params, samples, batch
            )
            elbo_samples = log_likelihood - log_posterior + log_prior
            return elbo_samples

        elbo_evaluation = elbo_samples(posterior_samples)
        concrete_evaluation = elbo_samples(concrete_samples)
        cv_evaluation = elbo_samples(cv_samples)

        obj = (lax.stop_gradient(elbo_evaluation - cv_coeff * cv_evaluation)) * log_posterior
        obj += cv_coeff * (concrete_evaluation - cv_evaluation)
        return - np.mean(obj, axis=0) / batch.shape[0]

    return rebar_objective


def generate_tunable_rebar_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder)

    def tunable_rebar_objective(encoder_params, decoder_params, batch, prng_key, cv_coeff=1.0,
                                log_temperature=0.5, num_samples=1):
        """
        Computes the REBAR objective function of a discrete VAE.

        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param cv_coeff: control variate coefficient
        :param log_temperature: log_temperature parameter for rebar control variate.
        :param num_samples: number of samples
        """
        temperature = np.exp(log_temperature)
        posterior_params = encoder(encoder_params, batch)  # BxD

        def concrete(x):
            return jax.nn.sigmoid(x * temperature)

        # Sampling
        sampling_key, control_variate_key = random.split(prng_key)

        # posterior_samples = bernoulli.sample(posterior_params, sampling_key, num_samples)
        relaxed_samples = binary_relaxed.sample(posterior_params, sampling_key, num_samples)

        posterior_samples = np.heaviside(relaxed_samples, 0)
        # concrete_samples = jax.nn.sigmoid(relaxed_samples * temperature)

        conditional_relaxed_samples = binary_relaxed.conditional_sample(posterior_params,
                                                                        posterior_samples, control_variate_key)
        # cv_samples = jax.nn.sigmoid(conditional_relaxed_samples * temperature)
        # frozen_cv_samples = binary_relaxed.conditional_sample(lax.stop_gradient(posterior_params), posterior_samples,
        #                                                       control_variate_key)
        # frozen_cv_samples = jax.nn.sigmoid(lax.stop_gradient(conditional_relaxed_samples) * temperature)

        log_posterior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(posterior_samples, posterior_params),
            axis=(1, 2)
        )

        def elbo_samples(samples, stop_grads=False):
            log_prior = np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(samples, 0.5 * np.ones((batch.shape[0],
                                                                                  posterior_params.shape[-1]))),
                axis=(1, 2)
            )  # SxBxD

            if stop_grads:
                log_posterior = np.sum(
                    vmap(bernoulli.logpmf, in_axes=(0, None))(samples, lax.stop_gradient(posterior_params)),
                    axis=(1, 2)
                )
            else:
                log_posterior = np.sum(
                    vmap(bernoulli.logpmf, in_axes=(0, None))(samples, posterior_params),
                    axis=(1, 2)
                )
            log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(decoder_params, samples, batch)
            elbo_samples = log_likelihood - log_posterior + log_prior
            return elbo_samples

        elbo_evaluation = elbo_samples(posterior_samples)
        concrete_evaluation = elbo_samples(concrete(relaxed_samples))
        cv_evaluation = elbo_samples((concrete(conditional_relaxed_samples)))
        frozen_cv_evaluation = elbo_samples(concrete(lax.stop_gradient(conditional_relaxed_samples)), True)

        obj = (lax.stop_gradient(elbo_evaluation) - cv_coeff * frozen_cv_evaluation) * log_posterior

        obj += cv_coeff * (concrete_evaluation - cv_evaluation)

        return - np.mean(obj, axis=0) / batch.shape[0]

    def gradient_variance(grad_fn, hyperparameters):
        """
        Variance of Gradient w.r.t. hyperparameters
        :param grad_fn: Gradient computing function w.r.t. tunable hyperparameters
        :param hyperparameters: cv_coeff and log_temperature (dict)
        """
        gradients = grad_fn(**hyperparameters)
        var = tree_reduce(lambda x, y: x + y, tree_map(lambda x: np.sum(x ** 2), gradients))
        return var
    return tunable_rebar_objective, gradient_variance


def generate_relax_objective(encoder, decoder, surrogate):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder)

    def relax_objective(encoder_params, decoder_params, surrogate_params, batch, prng_key, num_samples=1):
        """
        Computes the REBAR objective function of a discrete VAE.

        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param surrogate_params: surrogate parameters (list)
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param num_samples: number of samples
        """
        posterior_params = encoder(encoder_params, batch)  # BxD

        # Sampling
        sampling_key, conditional_sampling_key = random.split(prng_key)

        # posterior_samples = bernoulli.sample(posterior_params, sampling_key, num_samples)
        unconditional_samples = binary_relaxed.sample(posterior_params, sampling_key, num_samples)
        posterior_samples = np.heaviside(unconditional_samples, 0)
        conditional_samples = binary_relaxed.conditional_sample(posterior_params,
                                                                posterior_samples, conditional_sampling_key)

        log_posterior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(posterior_samples, posterior_params),
            axis=(1, 2)
        )

        def elbo_samples(samples):
            log_prior = np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(samples, 0.5 * np.ones((batch.shape[0],
                                                                                  posterior_params.shape[-1]))),
                axis=(1, 2)
            )  # SxBxD
            log_posterior = np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(samples, posterior_params),
                axis=(1, 2)
            )
            log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(decoder_params, samples, batch)
            elbo_samples = log_likelihood - log_posterior + log_prior
            return elbo_samples

        elbo_evaluation = elbo_samples(posterior_samples)

        obj = (
                      lax.stop_gradient(elbo_evaluation) -
                      np.sum(surrogate(surrogate_params, lax.stop_gradient(conditional_samples)), axis=1).squeeze()
               ) * log_posterior

        obj += np.sum(
            surrogate(surrogate_params, unconditional_samples) - surrogate(surrogate_params, conditional_samples),
            axis=1).squeeze()

        return - np.mean(obj, axis=0) / batch.shape[0]

    def gradient_variance(grad_fn, hyperparameters):
        """"
        Variance of the gradient w.r.t. surrogate parameters.

        :param grad_fn: Gradient computing function w.r.t. tunable hyperparameters
        :param hyperparameters: parameters of the RELAX surrogate (list)
        """
        gradients = grad_fn(surrogate_params=hyperparameters['surrogate_params'])
        var = tree_reduce(lambda x, y: x + y, tree_map(lambda x: np.sum(x ** 2), gradients))
        return var
    return relax_objective, gradient_variance


def generate_controlled_score_function_objective(encoder, decoder):
    objective = generate_score_function_objective(encoder, decoder)
    objective.__name__ = 'controlled_score_function_objective'

    def log_posterior(encoder_params, batch, prng_key, num_samples=1):
        posterior_params = encoder(encoder_params, batch)  # BxD
        posterior_samples = bernoulli.sample(posterior_params, prng_key, num_samples)

        log_posterior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(lax.stop_gradient(posterior_samples), posterior_params),
            axis=(1, 2)
        )
        return np.mean(log_posterior, axis=0)

    score = jax.grad(log_posterior)

    return objective, score


def generate_arm_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder)

    def arm_objective(encoder_params, decoder_params, batch, prng_key, num_samples=1):
        """
        Computes the ARM objective of a discrete VAE.
        The gradient of this objective matches the core function gradient of the ELBO.
        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param num_samples: number of samples
        """

        posterior_params = encoder(encoder_params, batch)  # BxD

        uniform_samples = random.uniform(key=prng_key, shape=(num_samples, *posterior_params.shape))
        antithetic_samples = 1 - uniform_samples

        uniform_condition = uniform_samples <= posterior_params
        antithetic_condition = antithetic_samples <= posterior_params

        def elbo_samples(x):
            log_prior = np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(x, 0.5 * np.ones((batch.shape[0],
                                                                            posterior_params.shape[-1]))),
                axis=(1, 2)
            )  # SxBxD
            log_posterior = np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(x, posterior_params), 
                axis=(1, 2)
            )
            log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(decoder_params, x, batch)
            elbo = log_likelihood - log_posterior + log_prior
            return elbo

        loss = lax.stop_gradient(elbo_samples(antithetic_condition) - elbo_samples(uniform_condition))
        loss = loss * np.sum(posterior_params * (uniform_samples - 0.5))
        return - np.mean(loss, axis=0) / batch.shape[0]
    return arm_objective


def generate_relax_plus_rebar_objective(encoder, decoder, surrogate):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder)

    def relax_plus_rebar_objective(encoder_params, decoder_params, surrogate_params, log_temperature,
                                   batch, prng_key, num_samples=1):
        """
        Computes the REBAR objective function of a discrete VAE.

        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param surrogate_params: surrogate parameters (list)
        :param log_temperature: log of inverse temperature
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param num_samples: number of samples
        """
        temperature = np.exp(log_temperature)

        def concrete(x):
            return jax.nn.sigmoid(x * temperature)

        posterior_params = encoder(encoder_params, batch)  # BxD

        # Sampling
        sampling_key, conditional_sampling_key = random.split(prng_key)

        # posterior_samples = bernoulli.sample(posterior_params, sampling_key, num_samples)
        unconditional_samples = binary_relaxed.sample(posterior_params, sampling_key, num_samples)
        posterior_samples = np.heaviside(unconditional_samples, 0)
        conditional_samples = binary_relaxed.conditional_sample(posterior_params,
                                                                posterior_samples, conditional_sampling_key)

        log_posterior = np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(posterior_samples, posterior_params),
            axis=(1, 2)
        )

        def elbo_samples(samples, stop_grads=False):
            log_prior = np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(samples, 0.5 * np.ones((batch.shape[0],
                                                                                  posterior_params.shape[-1]))),
                axis=(1, 2)
            )  # SxBxD

            if stop_grads:
                log_posterior = np.sum(
                    vmap(bernoulli.logpmf, in_axes=(0, None))(samples, lax.stop_gradient(posterior_params)),
                    axis=(1, 2)
                )
            else:
                log_posterior = np.sum(
                    vmap(bernoulli.logpmf, in_axes=(0, None))(samples, posterior_params),
                    axis=(1, 2)
                )
            log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(decoder_params, samples, batch)
            elbo_samples = log_likelihood - log_posterior + log_prior
            return elbo_samples

        elbo_evaluation = elbo_samples(posterior_samples)
        unconditional_evaluation = elbo_samples(concrete(unconditional_samples))
        conditional_evaluation = elbo_samples(concrete(conditional_samples))
        frozen_conditional_evaluation = elbo_samples(concrete(lax.stop_gradient(conditional_samples)), True)

        obj = (
                      lax.stop_gradient(elbo_evaluation) -
                      frozen_conditional_evaluation -
                      np.sum(surrogate(surrogate_params, lax.stop_gradient(conditional_samples)), axis=1).squeeze()
              ) * log_posterior

        obj += unconditional_evaluation + np.sum(surrogate(surrogate_params, unconditional_samples), axis=1).squeeze()
        obj -= conditional_evaluation + np.sum(surrogate(surrogate_params, conditional_samples), axis=1).squeeze()

        return - np.mean(obj, axis=0) / batch.shape[0]

    def gradient_variance(grad_fn, hyperparameters):
        """"
        Variance of the gradient w.r.t. surrogate parameters.

        :param grad_fn: Gradient computing function w.r.t. tunable hyperparameters
        :param hyperparameters: parameters of the RELAX surrogate (list)
        """
        gradients = grad_fn(surrogate_params=hyperparameters['surrogate_params'],
                            log_temperature=hyperparameters['log_temperature'])
        var = tree_reduce(lambda x, y: x + y, tree_map(lambda x: np.sum(x ** 2), gradients))
        return var
    return relax_plus_rebar_objective, gradient_variance
