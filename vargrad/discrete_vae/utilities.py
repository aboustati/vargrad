import os, time

import jax
import jax.numpy as np
import matplotlib.pyplot as plt

from functools import partial

from jax import grad, jit, lax, random, tree_map
from jax.experimental import optimizers

from tqdm import trange

from ..datasets import mnist, omniglot


def fori_loop(lower, upper, body_fun, init_val):
    """
    For dubgging only
    """
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


_TUNABLE_OBJECTIVES = ['tunable_rebar_objective', 'relax_objective', 'relax_plus_rebar_objective']


@jit
def compute_cv_coeff(gradient, control_variate):
    tol = 1e-7
    numerator = jax.tree_multimap(lambda x, y: x * y, gradient, control_variate)
    denominator = jax.tree_map(lambda x: x ** 2, control_variate)
    return jax.tree_multimap(lambda x, y: x / (y + tol), numerator, denominator)


@jit
def update_cv_coeff(history, new_coeff):
    return jax.tree_multimap(lambda x, y: 0.9 * x + 0.1 * y, history, new_coeff)


def image_grid(nrow, ncol, imagevecs, imshape):
    """
    Reshape a stack of image vectors into an image grid for plotting.
    """
    images = iter(imagevecs.reshape((-1,) + imshape))
    return np.vstack([np.hstack([next(images).T for _ in range(ncol)][::-1])
                      for _ in range(nrow)]).T


def image_sample(rng, decoder, nrow, ncol):
    """
    Sample images from the generative model.

    :param rng: PRNG key
    :param decoder: tuple of the decoding function and decoder parameters
    :param nrow: number of rows
    :param ncol: number of columns
    """
    decode, decoder_params = decoder
    code_rng, img_rng = random.split(rng)
    preds = decode(decoder_params, random.bernoulli(code_rng, shape=(nrow * ncol, 200)))
    sampled_images = random.bernoulli(img_rng, preds)
    return image_grid(nrow, ncol, sampled_images, (28, 28))


def two_layer_image_sample(rng, decoder, nrow, ncol):
    """
    Sample images from the generative model.

    :param rng: PRNG key
    :param decoder: tuple of the decoding function and decoder parameters
    :param nrow: number of rows
    :param ncol: number of columns
    """
    decode, decoder_params = decoder
    decode1, decode2 = decode
    decoder_params1, decoder_params2 = decoder_params
    code1_rng, code2_rng, img_rng = random.split(rng, num=3)

    inner_layer = decode2(decoder_params2, random.bernoulli(code2_rng, shape=(nrow * ncol, 200)))
    outer_layer = decode1(decoder_params1, random.bernoulli(code1_rng, inner_layer))
    sampled_images = random.bernoulli(img_rng, outer_layer)
    return image_grid(nrow, ncol, sampled_images, (28, 28))


def setup_experiment(objective_fn, elbo_fn, encoder_init, decoder_init, decoder,
                     config, dataset='mnist', two_layer=False, **kwargs):
    objective_name = objective_fn.__name__

    _is_controlled = (objective_name == 'controlled_score_function')
    _is_tunable = (objective_name in _TUNABLE_OBJECTIVES)

    nrow, ncol = 10, 10  # sampled image grid size

    test_rng = random.PRNGKey(config['test_seed'])  # fixed prng key for evaluation

    if not os.path.exists(config['image_path']):
        os.makedirs(config['image_path'])

    imfile = os.path.join(config['image_path'], objective_name + f'_{dataset}_' + "vae_{:03d}.png")
    batch_size = config['batch_size']
    num_samples = config['num_samples']

    if dataset == 'mnist':
        train_images, test_images = mnist(permute_train=True)
    elif dataset == 'omniglot':
        train_images, test_images = omniglot(permute_train=True)
    num_complete_batches, leftover = divmod(train_images.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)

    if two_layer:
        enc1_init_rng, enc2_init_rng, dec1_init_rng, dec2_init_rng = random.split(
            random.PRNGKey(config['init_seed']), num=4
        )
        _, init_encoder1_params = encoder_init[0](enc1_init_rng, (batch_size, 28 * 28))
        _, init_encoder2_params = encoder_init[1](enc2_init_rng, (batch_size, 200))
        _, init_decoder1_params = decoder_init[0](dec1_init_rng, (batch_size, 200))
        _, init_decoder2_params = decoder_init[1](dec2_init_rng, (batch_size, 200))
        init_encoder_params = (init_encoder1_params, init_encoder2_params)
        init_decoder_params = (init_decoder1_params, init_decoder2_params)
        init_params = (init_encoder_params, init_decoder_params)
    else:
        enc_init_rng, dec_init_rng = random.split(random.PRNGKey(config['init_seed']))
        _, init_encoder_params = encoder_init(enc_init_rng, (batch_size, 28 * 28))
        _, init_decoder_params = decoder_init(dec_init_rng, (batch_size, 200))
        init_params = (init_encoder_params, init_decoder_params)

    opt_init, opt_update, get_params = optimizers.adam(config['learning_rate'])

    if _is_tunable:
        cv_opt_init, cv_opt_update, cv_get_params = optimizers.adam(10 * config['learning_rate'])

        if objective_name == 'relax_objective' or objective_name == 'relax_plus_rebar_objective':
            surrogate_init_rng = random.fold_in(random.PRNGKey(config['init_seed']), 42)
            if two_layer:
                _, init_surrogate_params = kwargs['surrogate_init'](surrogate_init_rng, (num_samples, batch_size, 400))
            else:
                _, init_surrogate_params = kwargs['surrogate_init'](surrogate_init_rng, (num_samples, batch_size, 200))
            init_hyperparameters = {'surrogate_params': init_surrogate_params}
            if objective_name == 'relax_plus_rebar_objective':
                init_hyperparameters['log_temperature'] = 0.7

        elif objective_name == 'tunable_rebar_objective':
            init_hyperparameters = {
                'cv_coeff': 0.5,
                'log_temperature': 0.7
            }

    def fetch_batch(i, images):
        i = i % num_batches
        batch = lax.dynamic_slice_in_dim(images, i * batch_size, batch_size)
        return batch

    if _is_tunable:
        @jit
        def run_epoch(rng, opt_state, cv_opt_state):
            def body_fun(i, args):
                opt_state, cv_opt_state = args
                params = get_params(opt_state)
                elbo_rng = random.fold_in(rng, i)
                batch = fetch_batch(i, train_images)

                hyperparameters = cv_get_params(cv_opt_state)

                grad_fn = partial(grad(objective_fn, argnums=0), *params, batch=batch,
                                  prng_key=elbo_rng, num_samples=num_samples)

                encoder_grad = grad_fn(**hyperparameters)
                decoder_grad = grad(elbo_fn, argnums=1)(*params, batch, elbo_rng, num_samples)
                g = (encoder_grad, decoder_grad)

                cv_grad = grad(kwargs['support_fn'], argnums=1)(grad_fn, hyperparameters)

                return opt_update(i, g, opt_state), cv_opt_update(i, cv_grad, cv_opt_state)
            return lax.fori_loop(0, num_batches, body_fun, (opt_state, cv_opt_state))

    elif _is_controlled:
        @jit
        def run_epoch(rng, opt_state, cv_history):
            def body_fun(i, args):
                opt_state, cv_history = args
                params = get_params(opt_state)
                elbo_rng, data_rng, cv_rng = random.split(random.fold_in(rng, i), num=3)
                batch = fetch_batch(i, train_images)
                encoder_grad = grad(objective_fn)(*params, batch, elbo_rng, num_samples)
                decoder_grad = grad(elbo_fn, argnums=1)(*params, batch, elbo_rng, num_samples)
                control_variate = kwargs['support_fn'](params[0], batch, elbo_rng, num_samples)

                # Old computation of CV coeff as exponentially weighted moving average
                cv_coeff = compute_cv_coeff(encoder_grad, control_variate)
                cv_history = update_cv_coeff(cv_history, cv_coeff)
                encoder_grad = jax.tree_multimap(lambda x, y, z: x - y * z, encoder_grad, cv_history, control_variate)

                # coeff_encoder_grad = grad(objective_fn)(*params, batch, cv_rng, num_samples)
                # coeff_control_variate = kwargs['support_fn'](params[0], batch, cv_rng, num_samples)

                # cv_coeff = compute_cv_coeff(coeff_encoder_grad, coeff_control_variate)

                encoder_grad = jax.tree_multimap(lambda x, y, z: x - y * z, encoder_grad, cv_coeff, control_variate)

                g = (encoder_grad, decoder_grad)
                return opt_update(i, g, opt_state), cv_history

            return lax.fori_loop(0, num_batches, body_fun, (opt_state, cv_history))

    else:
        @jit
        def run_epoch(rng, opt_state):
            def body_fun(i, opt_state):
                params = get_params(opt_state)
                elbo_rng = random.fold_in(rng, i)
                batch = fetch_batch(i, train_images)
                encoder_grad = grad(objective_fn)(*params, batch=batch, prng_key=elbo_rng, num_samples=num_samples)
                decoder_grad = grad(elbo_fn, argnums=1)(*params, batch, elbo_rng, num_samples)
                g = (encoder_grad, decoder_grad)
                return opt_update(i, g, opt_state)
            return lax.fori_loop(0, num_batches, body_fun, opt_state)

    @jit
    def evaluate(opt_state):
        params = get_params(opt_state)
        elbo_rng = random.fold_in(test_rng, 1)
        test_elbo = elbo_fn(*params, test_images, elbo_rng, num_samples=5)
        return test_elbo

    if two_layer:
        @jit
        def generate_sample_images(opt_state):
            params = get_params(opt_state)
            image_rng = random.fold_in(test_rng, 2)
            sampled_images = two_layer_image_sample(image_rng, (decoder, params[1]), nrow, ncol)
            return sampled_images
    else:
        @jit
        def generate_sample_images(opt_state):
            params = get_params(opt_state)
            image_rng = random.fold_in(test_rng, 2)
            sampled_images = image_sample(image_rng, (decoder, params[1]), nrow, ncol)
            return sampled_images

    def training_loop(verbose=False):

        results_dict = {
            'test_elbo': [],
            'epoch_time': [],
        }
        state_dict = {
            'param_state': [],
            'hyperparam_state': []
        }
        key = random.PRNGKey(config['train_seed'])
        opt_state = opt_init(init_params)
        if _is_tunable:
            cv_opt_state = cv_opt_init(init_hyperparameters)
        elif _is_controlled:
            cv_history = jax.tree_multimap(lambda x: np.zeros_like(x), init_encoder_params)

        # Initial values
        test_elbo = evaluate(opt_state)
        results_dict['test_elbo'].append(test_elbo.tolist())
        results_dict['epoch_time'].append(0)

        for epoch in trange(config['num_epochs']):
            tic = time.time()
            key, _ = random.split(key)

            if _is_tunable:
                opt_state, cv_opt_state = run_epoch(key, opt_state, cv_opt_state)
            elif _is_controlled:
                opt_state, cv_history = run_epoch(key, opt_state, cv_history)
            else:
                opt_state = run_epoch(key, opt_state)

            test_elbo = evaluate(opt_state)
            test_elbo.block_until_ready()

            time_diff = time.time() - tic

            save_im = (epoch % 10 == 0 or epoch == 199)
            if save_im:
                sampled_images = generate_sample_images(opt_state)
                plt.imsave(imfile.format(epoch), sampled_images, cmap=plt.cm.gray)

            results_dict['test_elbo'].append(test_elbo.tolist())
            results_dict['epoch_time'].append(time_diff)

            if save_im:
                state_dict['param_state'].append(tree_map(lambda x: x.tolist(), get_params(opt_state)))
                if _is_tunable:
                    state_dict['hyperparam_state'].append(tree_map(lambda x: x.tolist(), cv_get_params(cv_opt_state)))
                elif _is_controlled:
                    state_dict['hyperparam_state'].append(tree_map(lambda x: x.tolist(), cv_history))

            if verbose:
                print("{: 3d} {} ({:.3f} sec)".format(epoch, test_elbo, time_diff))

        return results_dict, state_dict
    return training_loop
