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
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder[0])

    def negative_elbo(encoder_params, decoder_params, batch, prng_key, num_samples=1):
        """
        Computes the ELBO of a discrete VAE
        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param num_samples: number of samples
        """
        encoder1, encoder2 = encoder
        encoder_params1, encoder_params2 = encoder_params
        decoder1, decoder2 = decoder
        decoder_params1, decoder_params2 = decoder_params

        first_layer_key, second_layer_key = random.split(prng_key)

        # Outer layer latents
        first_layer_params = encoder1(encoder_params1, batch)  # BxD
        first_layer_samples = bernoulli.sample(first_layer_params, first_layer_key, num_samples)

        # Inner layer latents
        second_layer_params = encoder2(encoder_params2, first_layer_samples)
        second_layer_samples = bernoulli.sample(second_layer_params, second_layer_key, num_samples=1)[0, ...]

        # Inner layer prior
        log_prior = np.sum(
            bernoulli.logpmf(second_layer_samples, decoder2(decoder_params2, first_layer_samples)),
            axis=(1, 2)
        )  # SxBxD
        # Outer layer prior
        log_prior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, 0.5 * np.ones((batch.shape[0], 200))),
            axis=(1, 2)
        )  # SxBxD

        # Inner layer posterior
        log_posterior = np.sum(bernoulli.logpmf(second_layer_samples, second_layer_params), axis=(1, 2))
        # Outer layer posterior
        log_posterior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, first_layer_params),
            axis=(1, 2)
        )

        # Likelihood
        log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
            decoder_params1, first_layer_samples, batch
        )

        elbo_samples = log_likelihood - log_posterior + log_prior

        return - np.mean(elbo_samples, axis=0) / batch.shape[0]
    return negative_elbo


def generate_score_function_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder[0])

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

        encoder1, encoder2 = encoder
        encoder_params1, encoder_params2 = encoder_params
        decoder1, decoder2 = decoder
        decoder_params1, decoder_params2 = decoder_params

        first_layer_key, second_layer_key = random.split(prng_key)

        # Outer layer latents
        first_layer_params = encoder1(encoder_params1, batch)  # BxD
        first_layer_samples = bernoulli.sample(first_layer_params, first_layer_key, num_samples)

        # Inner layer latents
        second_layer_params = encoder2(encoder_params2, first_layer_samples)
        second_layer_samples = bernoulli.sample(second_layer_params, second_layer_key, num_samples=1)[0, ...]

        # Inner layer prior
        log_prior = np.sum(
            bernoulli.logpmf(second_layer_samples, decoder2(decoder_params2, first_layer_samples)),
            axis=(1, 2)
        )  # SxBxD
        # Outer layer prior
        log_prior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, 0.5 * np.ones((batch.shape[0], 200))),
            axis=(1, 2)
        )  # SxBxD

        # Inner layer posterior
        log_posterior = np.sum(bernoulli.logpmf(second_layer_samples, second_layer_params), axis=(1, 2))
        # Outer layer posterior
        log_posterior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, first_layer_params),
            axis=(1, 2)
        )

        # Likelihood
        log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
            decoder_params[0], first_layer_samples, batch
        )

        elbo_samples = log_likelihood - log_posterior + log_prior
        elbo_samples = lax.stop_gradient(elbo_samples) * log_posterior
        return - np.mean(elbo_samples, axis=0) / batch.shape[0]
    return score_function_objective


def generate_variance_loss_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder[0])

    def variance_loss_objective(encoder_params, decoder_params, batch, prng_key, num_samples=1):
        """
        Computes the variance loss objective of a discrete VAE.
        :param encoder_params: encoder parameters (list)
        :param decoder_params: decoder parameters (list)
        :param batch: batch of data (jax.numpy array)
        :param prng_key: PRNG key
        :param num_samples: number of samples
        """
        encoder1, encoder2 = encoder
        encoder_params1, encoder_params2 = encoder_params
        decoder1, decoder2 = decoder
        decoder_params1, decoder_params2 = decoder_params

        first_layer_key, second_layer_key = random.split(prng_key)

        # Outer layer latents
        first_layer_params = encoder1(encoder_params1, batch)  # BxD
        first_layer_samples = lax.stop_gradient(bernoulli.sample(first_layer_params, first_layer_key, num_samples))

        # Inner layer latents
        second_layer_params = encoder2(encoder_params2, first_layer_samples)
        second_layer_samples = lax.stop_gradient(
            bernoulli.sample(second_layer_params, second_layer_key, num_samples=1)[0, ...]
        )

        # Inner layer prior
        log_prior = np.sum(
            bernoulli.logpmf(second_layer_samples, decoder2(decoder_params2, first_layer_samples)),
            axis=(1, 2)
        )  # SxBxD
        # Outer layer prior
        log_prior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, 0.5 * np.ones((batch.shape[0], 200))),
            axis=(1, 2)
        )  # SxBxD

        # Inner layer posterior
        log_posterior = np.sum(bernoulli.logpmf(second_layer_samples, second_layer_params), axis=(1, 2))
        # Outer layer posterior
        log_posterior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, first_layer_params),
            axis=(1, 2)
        )

        # Likelihood
        log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
            decoder_params1, first_layer_samples, batch
        )

        elbo_samples = log_likelihood - log_posterior + log_prior

        return np.var(elbo_samples, axis=0, ddof=1) / batch.shape[0]
    return variance_loss_objective


def generate_concrete_relaxation_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder[0])

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

        encoder1, encoder2 = encoder
        encoder_params1, encoder_params2 = encoder_params
        decoder1, decoder2 = decoder
        decoder_params1, decoder_params2 = decoder_params

        first_layer_key, second_layer_key = random.split(prng_key)

        # Outer layer latents
        first_layer_params = encoder1(encoder_params1, batch)  # BxD
        first_layer_samples = binary_concrete.sample(first_layer_params, temperatures[1], first_layer_key, num_samples)

        # Inner layer latents
        second_layer_params = encoder2(encoder_params2, first_layer_samples)
        second_layer_samples = binary_concrete.sample(second_layer_params, temperatures[1],
                                                      second_layer_key, num_samples=1)[0, ...]

        # Inner layer prior
        log_prior = np.sum(
            binary_concrete.logpdf(
                second_layer_samples, decoder2(decoder_params2, first_layer_samples), temperatures[0]
            ),
            axis=(1, 2)
        )  # SxBxD
        # Outer layer prior
        log_prior += np.sum(
            vmap(lambda x, y: binary_concrete.logpdf(x, y, temperatures[0]), in_axes=(0, None))(
                first_layer_samples,
                0.5 * np.ones((batch.shape[0], 200))
            ),
            axis=(1, 2)
        )  # SxBxD

        # Inner layer posterior
        log_posterior = np.sum(
            binary_concrete.logpdf(second_layer_samples, second_layer_params, temperatures[1]),
            axis=(1, 2)
        )
        # Outer layer posterior
        log_posterior += np.sum(
            vmap(lambda x, y: binary_concrete.logpdf(x, y, temperatures[1]), in_axes=(0, None))(
                first_layer_samples, first_layer_params
            ),
            axis=(1, 2)
        )

        # Likelihood
        log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
            decoder_params1, first_layer_samples, batch
        )

        elbo_samples = log_likelihood - log_posterior + log_prior

        return - np.mean(elbo_samples, axis=0) / batch.shape[0]
    return concrete_relaxation_objective


def generate_rebar_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder[0])

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
        encoder1, encoder2 = encoder
        encoder_params1, encoder_params2 = encoder_params
        decoder1, decoder2 = decoder
        decoder_params1, decoder_params2 = decoder_params

        # Sampling
        sampling_key, control_variate_key = random.split(prng_key)
        first_layer_sampling_key, second_layer_sampling_key = random.split(sampling_key)
        first_layer_control_variate_key, second_layer_control_variate_key = random.split(control_variate_key)

        # Outer Layer
        first_layer_params = encoder1(encoder_params1, batch)  # BxD
        first_layer_relaxed_samples = binary_relaxed.sample(first_layer_params, first_layer_sampling_key, num_samples)
        first_layer_posterior_samples = np.heaviside(first_layer_relaxed_samples, 0)
        first_layer_concrete_samples = jax.nn.sigmoid(first_layer_relaxed_samples / temperature)
        first_layer_cv_samples = binary_concrete.conditional_sample(first_layer_params, first_layer_posterior_samples,
                                                                    temperature, first_layer_control_variate_key)

        # Inner Layer
        second_layer_params = encoder2(encoder_params2, first_layer_posterior_samples)
        second_layer_relaxed_samples = binary_relaxed.sample(
            second_layer_params, second_layer_sampling_key, num_samples=1
        )[0, ...]
        second_layer_posterior_samples = np.heaviside(second_layer_relaxed_samples, 0)
        second_layer_concrete_samples = jax.nn.sigmoid(second_layer_relaxed_samples / temperature)
        second_layer_cv_samples = binary_concrete.conditional_sample(
            second_layer_params, second_layer_posterior_samples, temperature, second_layer_control_variate_key
        )

        # Inner layer posterior
        log_posterior = np.sum(bernoulli.logpmf(second_layer_posterior_samples, second_layer_params), axis=(1, 2))
        # Outer layer posterior
        log_posterior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_posterior_samples, first_layer_params),
            axis=(1, 2)
        )

        def elbo_samples(first_layer_samples, second_layer_samples):
            # Inner layer prior
            log_prior = np.sum(
                bernoulli.logpmf(second_layer_samples, decoder2(decoder_params2, first_layer_samples)),
                axis=(1, 2)
            )  # SxBxD
            # Outer layer prior
            log_prior += np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, 0.5 * np.ones((batch.shape[0], 200))),
                axis=(1, 2)
            )  # SxBxD

            # Inner layer posterior
            log_posterior = np.sum(bernoulli.logpmf(second_layer_samples, second_layer_params), axis=(1, 2))
            # Outer layer posterior
            log_posterior += np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, first_layer_params),
                axis=(1, 2)
            )

            # Likelihood
            log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
                decoder_params1, first_layer_samples, batch
            )

            elbo_samples = log_likelihood - log_posterior + log_prior
            return elbo_samples

        elbo_evaluation = elbo_samples(first_layer_posterior_samples, second_layer_posterior_samples)
        concrete_evaluation = elbo_samples(first_layer_concrete_samples, second_layer_concrete_samples)
        cv_evaluation = elbo_samples(first_layer_cv_samples, second_layer_cv_samples)

        obj = (lax.stop_gradient(elbo_evaluation - cv_coeff * cv_evaluation)) * log_posterior
        obj += cv_coeff * (concrete_evaluation - cv_evaluation)
        return - np.mean(obj, axis=0) / batch.shape[0]

    return rebar_objective


def generate_tunable_rebar_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder[0])

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
        encoder1, encoder2 = encoder
        encoder_params1, encoder_params2 = encoder_params
        decoder1, decoder2 = decoder
        decoder_params1, decoder_params2 = decoder_params

        # Sampling
        sampling_key, control_variate_key = random.split(prng_key)
        first_layer_sampling_key, second_layer_sampling_key = random.split(sampling_key)
        first_layer_control_variate_key, second_layer_control_variate_key = random.split(control_variate_key)

        # Outer Layer
        first_layer_params = encoder1(encoder_params1, batch)  # BxD
        first_layer_relaxed_samples = binary_relaxed.sample(first_layer_params, first_layer_sampling_key, num_samples)
        first_layer_posterior_samples = np.heaviside(first_layer_relaxed_samples, 0)
        first_layer_concrete_samples = jax.nn.sigmoid(first_layer_relaxed_samples * temperature)
        first_layer_conditional_relaxed_samples = binary_relaxed.conditional_sample(
            first_layer_params, first_layer_posterior_samples, first_layer_control_variate_key
        )
        first_layer_cv_samples = jax.nn.sigmoid(first_layer_conditional_relaxed_samples * temperature)
        first_layer_frozen_cv_samples = jax.nn.sigmoid(
            lax.stop_gradient(first_layer_conditional_relaxed_samples) * temperature
        )

        # Inner Layer
        second_layer_params = encoder2(encoder_params2, first_layer_posterior_samples)
        second_layer_relaxed_samples = binary_relaxed.sample(
            second_layer_params, second_layer_sampling_key, num_samples=1
        )[0, ...]
        second_layer_posterior_samples = np.heaviside(second_layer_relaxed_samples, 0)
        second_layer_concrete_samples = jax.nn.sigmoid(second_layer_relaxed_samples * temperature)
        second_layer_conditional_relaxed_samples = binary_relaxed.conditional_sample(
            second_layer_params, second_layer_posterior_samples, second_layer_control_variate_key
        )
        second_layer_cv_samples = jax.nn.sigmoid(second_layer_conditional_relaxed_samples * temperature)
        second_layer_frozen_cv_samples = jax.nn.sigmoid(
            lax.stop_gradient(second_layer_conditional_relaxed_samples) * temperature
        )

        # Inner layer posterior
        log_posterior = np.sum(bernoulli.logpmf(second_layer_posterior_samples, second_layer_params), axis=(1, 2))
        # Outer layer posterior
        log_posterior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_posterior_samples, first_layer_params),
            axis=(1, 2)
        )

        def elbo_samples(first_layer_samples, second_layer_samples, stop_grads=False):
            # Inner layer prior
            log_prior = np.sum(
                bernoulli.logpmf(second_layer_samples, decoder2(decoder_params2, first_layer_samples)),
                axis=(1, 2)
            )  # SxBxD
            # Outer layer prior
            log_prior += np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, 0.5 * np.ones((batch.shape[0], 200))),
                axis=(1, 2)
            )  # SxBxD

            if stop_grads:
                # Inner layer posterior
                log_posterior = np.sum(
                    bernoulli.logpmf(second_layer_samples, lax.stop_gradient(second_layer_params)), axis=(1, 2)
                )
                # Outer layer posterior
                log_posterior += np.sum(
                    vmap(bernoulli.logpmf, in_axes=(0, None))(
                        first_layer_samples, lax.stop_gradient(first_layer_params)
                    ),
                    axis=(1, 2)
                )
            else:
                # Inner layer posterior
                log_posterior = np.sum(bernoulli.logpmf(second_layer_samples, second_layer_params), axis=(1, 2))
                # Outer layer posterior
                log_posterior += np.sum(
                    vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, first_layer_params),
                    axis=(1, 2)
                )

            # Likelihood
            log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
                decoder_params1, first_layer_samples, batch
            )

            elbo_samples = log_likelihood - log_posterior + log_prior
            return elbo_samples

        elbo_evaluation = elbo_samples(first_layer_posterior_samples, second_layer_posterior_samples)
        concrete_evaluation = elbo_samples(first_layer_concrete_samples, second_layer_concrete_samples)
        cv_evaluation = elbo_samples(first_layer_cv_samples, second_layer_cv_samples)
        frozen_cv_evaluation = elbo_samples(first_layer_frozen_cv_samples, second_layer_frozen_cv_samples, True)

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
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder[0])

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
        encoder1, encoder2 = encoder
        encoder_params1, encoder_params2 = encoder_params
        decoder1, decoder2 = decoder
        decoder_params1, decoder_params2 = decoder_params

        # Sampling
        sampling_key, control_variate_key = random.split(prng_key)
        first_layer_sampling_key, second_layer_sampling_key = random.split(sampling_key)
        first_layer_control_variate_key, second_layer_control_variate_key = random.split(control_variate_key)

        # Outer Layer
        first_layer_params = encoder1(encoder_params1, batch)  # BxD
        first_layer_relaxed_samples = binary_relaxed.sample(first_layer_params, first_layer_sampling_key, num_samples)
        first_layer_posterior_samples = np.heaviside(first_layer_relaxed_samples, 0)
        first_layer_conditional_relaxed_samples = binary_relaxed.conditional_sample(
            first_layer_params, first_layer_posterior_samples, first_layer_control_variate_key
        )

        # Inner Layer
        second_layer_params = encoder2(encoder_params2, first_layer_posterior_samples)
        second_layer_relaxed_samples = binary_relaxed.sample(
            second_layer_params, second_layer_sampling_key, num_samples=1
        )[0, ...]
        second_layer_posterior_samples = np.heaviside(second_layer_relaxed_samples, 0)
        second_layer_conditional_relaxed_samples = binary_relaxed.conditional_sample(
            second_layer_params, second_layer_posterior_samples, second_layer_control_variate_key
        )

        # Inner layer posterior
        log_posterior = np.sum(bernoulli.logpmf(second_layer_posterior_samples, second_layer_params), axis=(1, 2))
        # Outer layer posterior
        log_posterior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_posterior_samples, first_layer_params),
            axis=(1, 2)
        )

        def elbo_samples(first_layer_samples, second_layer_samples):
            # Inner layer prior
            log_prior = np.sum(
                bernoulli.logpmf(second_layer_samples, decoder2(decoder_params2, first_layer_samples)),
                axis=(1, 2)
            )  # SxBxD
            # Outer layer prior
            log_prior += np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, 0.5 * np.ones((batch.shape[0], 200))),
                axis=(1, 2)
            )  # SxBxD

            # Inner layer posterior
            log_posterior = np.sum(bernoulli.logpmf(second_layer_samples, second_layer_params), axis=(1, 2))
            # Outer layer posterior
            log_posterior += np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, first_layer_params),
                axis=(1, 2)
            )

            # Likelihood
            log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
                decoder_params1, first_layer_samples, batch
            )

            elbo_samples = log_likelihood - log_posterior + log_prior
            return elbo_samples

        elbo_evaluation = elbo_samples(first_layer_posterior_samples, second_layer_posterior_samples)

        relaxed_surrogate_inputs = np.concatenate([
            first_layer_relaxed_samples,
            second_layer_relaxed_samples
        ], axis=-1)

        conditional_relaxed_surrogate_inputs = np.concatenate([
            first_layer_conditional_relaxed_samples,
            second_layer_conditional_relaxed_samples
        ], axis=-1)

        obj = (
                      lax.stop_gradient(elbo_evaluation) -
                      np.sum(surrogate(
                          surrogate_params, lax.stop_gradient(conditional_relaxed_surrogate_inputs)
                      ), axis=1).squeeze()
               ) * log_posterior

        obj += np.sum(
            surrogate(surrogate_params, relaxed_surrogate_inputs) - surrogate(
                surrogate_params, conditional_relaxed_surrogate_inputs
            ),
            axis=1
        ).squeeze()

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
        encoder1, encoder2 = encoder
        encoder_params1, encoder_params2 = encoder_params

        first_layer_key, second_layer_key = random.split(prng_key)

        # Outer layer latents
        first_layer_params = encoder1(encoder_params1, batch)  # BxD
        first_layer_samples = bernoulli.sample(first_layer_params, first_layer_key, num_samples)

        # Inner layer latents
        second_layer_params = encoder2(encoder_params2, first_layer_samples)
        second_layer_samples = bernoulli.sample(second_layer_params, second_layer_key, num_samples=1)[0, ...]

        # Inner layer posterior
        log_posterior = np.sum(bernoulli.logpmf(second_layer_samples, second_layer_params), axis=(1, 2))
        # Outer layer posterior
        log_posterior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, first_layer_params),
            axis=(1, 2)
        )
        return np.mean(log_posterior, axis=0)

    score = jax.grad(log_posterior)

    return objective, score


def generate_arm_objective(encoder, decoder):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder[0])

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
        encoder1, encoder2 = encoder
        encoder_params1, encoder_params2 = encoder_params
        decoder1, decoder2 = decoder
        decoder_params1, decoder_params2 = decoder_params

        bernoulli_key, uniform_key1, uniform_key2 = random.split(prng_key, num=3)

        first_layer_params = encoder1(encoder_params1, batch)  # BxD
        first_layer_bernoulli_samples = bernoulli.sample(first_layer_params, bernoulli_key, num_samples)
        second_layer_params = encoder2(encoder_params2, first_layer_bernoulli_samples)

        first_layer_uniform_samples = random.uniform(key=uniform_key1, shape=(num_samples, *first_layer_params.shape))
        first_layer_antithetic_samples = 1 - first_layer_uniform_samples

        second_layer_uniform_samples = random.uniform(key=uniform_key2, shape=second_layer_params.shape)
        second_layer_antithetic_samples = 1 - second_layer_uniform_samples

        first_layer_uniform_condition = first_layer_uniform_samples <= first_layer_params
        first_layer_antithetic_condition = first_layer_antithetic_samples <= first_layer_params

        second_layer_uniform_condition = second_layer_uniform_samples <= second_layer_params
        second_layer_antithetic_condition = second_layer_antithetic_samples <= second_layer_params

        def elbo_samples(first_layer_samples, second_layer_samples):
            log_prior = np.sum(
                bernoulli.logpmf(second_layer_samples, decoder2(decoder_params2, first_layer_bernoulli_samples)),
                axis=(1, 2)
            )  # SxBxD
            log_prior += np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, 0.5 * np.ones((batch.shape[0], 200))),
                axis=(1, 2)
            )  # SxBxD

            log_posterior = np.sum(bernoulli.logpmf(second_layer_samples, second_layer_params), axis=(1, 2))
            log_posterior += np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, first_layer_params),
                axis=(1, 2)
            )

            log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
                decoder_params1, first_layer_samples, batch
            )

            elbo = log_likelihood - log_posterior + log_prior
            return elbo

        sample_elbo = elbo_samples(first_layer_uniform_condition, second_layer_uniform_condition)
        antithetic_sample_elbo = elbo_samples(
            first_layer_antithetic_condition, second_layer_antithetic_condition
        )

        loss = lax.stop_gradient(antithetic_sample_elbo - sample_elbo)
        loss = loss * (
                np.sum(first_layer_params * (first_layer_uniform_samples - 0.5)) +
                np.sum(second_layer_params * (second_layer_uniform_samples - 0.5))
        )

        return - np.mean(loss, axis=0) / batch.shape[0]
    return arm_objective


def generate_relax_plus_rebar_objective(encoder, decoder, surrogate):
    bernoulli_log_likelihood = _bernoulli_log_likelihood(decoder[0])

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

        encoder1, encoder2 = encoder
        encoder_params1, encoder_params2 = encoder_params
        decoder1, decoder2 = decoder
        decoder_params1, decoder_params2 = decoder_params

        # Sampling
        sampling_key, control_variate_key = random.split(prng_key)
        first_layer_sampling_key, second_layer_sampling_key = random.split(sampling_key)
        first_layer_control_variate_key, second_layer_control_variate_key = random.split(control_variate_key)

        # Outer Layer
        first_layer_params = encoder1(encoder_params1, batch)  # BxD
        first_layer_relaxed_samples = binary_relaxed.sample(first_layer_params, first_layer_sampling_key, num_samples)
        first_layer_posterior_samples = np.heaviside(first_layer_relaxed_samples, 0)
        first_layer_conditional_relaxed_samples = binary_relaxed.conditional_sample(
            first_layer_params, first_layer_posterior_samples, first_layer_control_variate_key
        )

        # Inner Layer
        second_layer_params = encoder2(encoder_params2, first_layer_posterior_samples)
        second_layer_relaxed_samples = binary_relaxed.sample(
            second_layer_params, second_layer_sampling_key, num_samples=1
        )[0, ...]
        second_layer_posterior_samples = np.heaviside(second_layer_relaxed_samples, 0)
        second_layer_conditional_relaxed_samples = binary_relaxed.conditional_sample(
            second_layer_params, second_layer_posterior_samples, second_layer_control_variate_key
        )

        # Inner layer posterior
        log_posterior = np.sum(bernoulli.logpmf(second_layer_posterior_samples, second_layer_params), axis=(1, 2))
        # Outer layer posterior
        log_posterior += np.sum(
            vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_posterior_samples, first_layer_params),
            axis=(1, 2)
        )

        def elbo_samples(first_layer_samples, second_layer_samples, stop_grads=False):
            # Inner layer prior
            log_prior = np.sum(
                bernoulli.logpmf(second_layer_samples, decoder2(decoder_params2, first_layer_samples)),
                axis=(1, 2)
            )  # SxBxD
            # Outer layer prior
            log_prior += np.sum(
                vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, 0.5 * np.ones((batch.shape[0], 200))),
                axis=(1, 2)
            )  # SxBxD

            if stop_grads:
                # Inner layer posterior
                log_posterior = np.sum(
                    bernoulli.logpmf(second_layer_samples, lax.stop_gradient(second_layer_params)), axis=(1, 2)
                )
                # Outer layer posterior
                log_posterior += np.sum(
                    vmap(bernoulli.logpmf, in_axes=(0, None))(
                         first_layer_samples, lax.stop_gradient(first_layer_params)
                    ),
                    axis=(1, 2)
                )
            else:
                # Inner layer posterior
                log_posterior = np.sum(bernoulli.logpmf(second_layer_samples, second_layer_params), axis=(1, 2))
                # Outer layer posterior
                log_posterior += np.sum(
                    vmap(bernoulli.logpmf, in_axes=(0, None))(first_layer_samples, first_layer_params),
                    axis=(1, 2)
                )

            # Likelihood
            log_likelihood = vmap(bernoulli_log_likelihood, in_axes=(None, 0, None))(
                decoder_params1, first_layer_samples, batch
            )

            elbo_samples = log_likelihood - log_posterior + log_prior
            return elbo_samples

        elbo_evaluation = elbo_samples(first_layer_posterior_samples, second_layer_posterior_samples)
        unconditional_evaluation = elbo_samples(
            concrete(first_layer_relaxed_samples), concrete(second_layer_relaxed_samples)
        )
        conditional_evaluation = elbo_samples(
            concrete(first_layer_conditional_relaxed_samples), concrete(second_layer_conditional_relaxed_samples)
        )
        frozen_conditional_evaluation = elbo_samples(
            concrete(lax.stop_gradient(first_layer_conditional_relaxed_samples)),
            concrete(lax.stop_gradient(second_layer_conditional_relaxed_samples)),
            stop_grads=True
        )

        relaxed_surrogate_inputs = np.concatenate([
            first_layer_relaxed_samples,
            second_layer_relaxed_samples
        ], axis=-1)

        conditional_relaxed_surrogate_inputs = np.concatenate([
            first_layer_conditional_relaxed_samples,
            second_layer_conditional_relaxed_samples
        ], axis=-1)

        obj = (
                      lax.stop_gradient(elbo_evaluation) - frozen_conditional_evaluation -
                      np.sum(surrogate(
                          surrogate_params, lax.stop_gradient(conditional_relaxed_surrogate_inputs)
                      ), axis=1).squeeze()
              ) * log_posterior

        obj += unconditional_evaluation + np.sum(surrogate(surrogate_params, relaxed_surrogate_inputs),
                                                 axis=1).squeeze()
        obj -= conditional_evaluation + np.sum(surrogate(surrogate_params, conditional_relaxed_surrogate_inputs),
                                               axis=1).squeeze()

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
