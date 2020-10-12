import os
import json
import random
from glob import glob
from itertools import product

from jax.experimental import stax
from jax.experimental.stax import Dense, Sigmoid, Relu

from vargrad_experiments.paths import CONFIG_PATH, RESULTS_PATH
from vargrad_experiments.discrete_vae.config import generate_single_sample_run_config
from vargrad_experiments.discrete_vae.instance import Instance, RelaxInstance

LOSSES = ['variance_loss']

MODEL = 'omniglot_one_layer_deep'

LOG_PATH = os.path.join(RESULTS_PATH, MODEL, 'log.txt')


def _create_networks():
    encoder1_init, encode1 = stax.serial(Dense(200), Sigmoid)

    encoder2_init, encode2 = stax.serial(Dense(200), Sigmoid)

    decoder2_init, decode2 = stax.serial(Dense(200), Sigmoid)

    decoder1_init, decode1 = stax.serial(Dense(28 * 28), Sigmoid)

    encoder = (encode1, encode2)
    encoder_init = (encoder1_init, encoder2_init)
    decoder = (decode1, decode2)
    decoder_init = (decoder1_init, decoder2_init)

    return encoder, encoder_init, decoder, decoder_init


class RunInstance(Instance):
    def create_networks(self):
        return _create_networks()


class RunRelaxInstance(RelaxInstance):
    def create_networks(self):
        return _create_networks()

    def create_surrogate(self):
        surrogate_init, surrogate = stax.serial(
            Dense(200), Relu,
            Dense(200), Relu,
            Dense(200), Relu,
            Dense(1)
        )
        return surrogate, surrogate_init


if __name__ == '__main__':
    generate_single_sample_run_config(MODEL)

    files = []
    start_dir = os.path.join(CONFIG_PATH, MODEL)
    pattern = "*.json"

    for dirs, _, _ in os.walk(start_dir):
        files.extend(glob(os.path.join(dirs, pattern)))

    runs = list(product(LOSSES, files))
    random.shuffle(runs)

    for loss, file in runs:
        with open(file, 'r') as f:
            config = json.load(f)

        print(f"Running {loss} from {file}")
        try:
            for s in [10, 15]:
                config['num_samples'] = s
                exp = RunInstance(os.path.join(RESULTS_PATH, MODEL), config=config, loss=loss, dataset='omniglot',
                                  two_layer=True)
                exp.run_and_save(prefix=f'num_samples_{s}')
                print("Run finished")
        except Exception as e:
            with open(LOG_PATH, 'a') as f:
                f.write('#' * 60 + '\n')
                f.write(str(e) + '\n')
                f.write(loss + '\n')
                f.write(file + '\n')
                f.write('#' * 60 + '\n')

            print(e)
            continue
