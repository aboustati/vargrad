import os
import json

from copy import deepcopy

from vargrad.discrete_vae.config import DEFAULT_CONFIG
from ..paths import CONFIG_PATH, RESULTS_PATH

EXPERIMENT_SEEDS = [192873, 1634, 29837, 1519, 63094]
LEARNING_RATES = [1e-3, 5e-4, 1e-4]


def change_training_seed(config_dict, new_seed):
    new_config = deepcopy(config_dict)
    new_config['train_seed'] = new_seed
    new_config['init_seed'] = new_seed * 3
    return new_config


def change_learning_rate(config_dict, new_learning_rate):
    new_config = deepcopy(config_dict)
    new_config['learning_rate'] = new_learning_rate
    return new_config


def change_path(config_dict, new_path):
    new_config = deepcopy(config_dict)
    new_config['image_path'] = new_path
    return new_config


def change_num_samples(config_dict, new_num_samples):
    new_config = deepcopy(config_dict)
    new_config['num_samples'] = new_num_samples
    return new_config


def change_batch_size(config_dict, new_batch_size):
    new_config = deepcopy(config_dict)
    new_config['batch_size'] = new_batch_size
    return new_config


def change_num_epochs(config_dict, new_num_epochs):
    new_config = deepcopy(config_dict)
    new_config['num_epochs'] = new_num_epochs
    return new_config


def generate_config(model):

    dvae_config_path = os.path.join(CONFIG_PATH, model)

    for learning_rate in LEARNING_RATES:
        config_path = os.path.join(dvae_config_path, f'learning_rate_{learning_rate}')
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        for i, seed in enumerate(EXPERIMENT_SEEDS):
            config = change_training_seed(DEFAULT_CONFIG, seed)
            config = change_learning_rate(config, learning_rate)
            config = change_path(config,
                                 os.path.join(RESULTS_PATH, f'{model}/learning_rate_{learning_rate}/run_{i}'))
            with open(os.path.join(config_path, f'config_{i}.json'), 'w') as f:
                json.dump(config, f)


def generate_single_sample_run_config(model):

    dvae_config_path = os.path.join(CONFIG_PATH, model)

    for learning_rate in LEARNING_RATES:
        config_path = os.path.join(dvae_config_path, f'learning_rate_{learning_rate}')
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        for i, seed in enumerate(EXPERIMENT_SEEDS):
            config = change_training_seed(DEFAULT_CONFIG, seed)
            config = change_learning_rate(config, learning_rate)
            config = change_num_samples(config, 1)
            config = change_num_epochs(config, 200)
            config = change_path(config,
                                 os.path.join(
                                     RESULTS_PATH, f'{model}/learning_rate_{learning_rate}/run_{i}'
                                 )
                                 )
            with open(os.path.join(config_path, f'config_{i}.json'), 'w') as f:
                json.dump(config, f)


def generate_very_deep_run_config(model):

    dvae_config_path = os.path.join(CONFIG_PATH, model)

    for learning_rate in LEARNING_RATES:
        config_path = os.path.join(dvae_config_path, f'learning_rate_{learning_rate}')
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        for i, seed in enumerate(EXPERIMENT_SEEDS[:2]):
            config = change_training_seed(DEFAULT_CONFIG, seed)
            config = change_learning_rate(config, learning_rate)
            config = change_num_samples(config, 1)
            config = change_path(config,
                                 os.path.join(
                                     RESULTS_PATH, f'{model}/learning_rate_{learning_rate}/run_{i}'
                                 )
                                 )
            with open(os.path.join(config_path, f'config_{i}.json'), 'w') as f:
                json.dump(config, f)


def generate_long_run_config(model):

    dvae_config_path = os.path.join(CONFIG_PATH, model)

    for learning_rate in LEARNING_RATES[:1]:
        config_path = os.path.join(dvae_config_path, f'learning_rate_{learning_rate}')
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        for i, seed in enumerate(EXPERIMENT_SEEDS[:2]):
            config = change_training_seed(DEFAULT_CONFIG, seed)
            config = change_learning_rate(config, learning_rate)
            config = change_num_samples(config, 1)
            config = change_num_epochs(config, 200)
            config = change_path(config,
                                 os.path.join(
                                     RESULTS_PATH, f'{model}/learning_rate_{learning_rate}/run_{i}'
                                 )
                                 )
            with open(os.path.join(config_path, f'config_{i}.json'), 'w') as f:
                json.dump(config, f)
