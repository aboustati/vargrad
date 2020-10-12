import json

DEFAULT_CONFIG = {
    'image_path': '/tmp/discrete_vae',
    'test_seed': 100,
    'train_seed': 42,
    'init_seed': 0,
    'num_samples': 5,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'num_epochs': 100,
}

if __name__ == '__main__':
    with open('../../configs/discrete_vae_default_config.json', 'w') as f:
        json.dump(DEFAULT_CONFIG, f)
