import json
import os

from abc import ABC, abstractmethod

import vargrad.discrete_vae.objectives as objectives
import vargrad.discrete_vae.two_layer_objectives as two_layer_objectives
from vargrad.discrete_vae.utilities import setup_experiment

LOSSES = {
    'score_function': objectives.generate_score_function_objective,
    'variance_loss': objectives.generate_variance_loss_objective,
    'concrete_relaxation': objectives.generate_concrete_relaxation_objective,
    'rebar': objectives.generate_rebar_objective,
    'relax': objectives.generate_relax_objective,
    'tunable_rebar': objectives.generate_tunable_rebar_objective,
    'controlled_score_function': objectives.generate_controlled_score_function_objective,
    'arm': objectives.generate_arm_objective,
    'relax_plus_rebar': objectives.generate_relax_plus_rebar_objective
}

TWO_LAYER_LOSSES = {
    'score_function': two_layer_objectives.generate_score_function_objective,
    'variance_loss': two_layer_objectives.generate_variance_loss_objective,
    'concrete_relaxation': two_layer_objectives.generate_concrete_relaxation_objective,
    'rebar': two_layer_objectives.generate_rebar_objective,
    'relax': two_layer_objectives.generate_relax_objective,
    'tunable_rebar': two_layer_objectives.generate_tunable_rebar_objective,
    'controlled_score_function': two_layer_objectives.generate_controlled_score_function_objective,
    'arm': two_layer_objectives.generate_arm_objective,
    'relax_plus_rebar': two_layer_objectives.generate_relax_plus_rebar_objective
}


class Instance(ABC):
    def __init__(self, output_path, config, loss, dataset='mnist', two_layer=False):
        self.output_path = output_path
        self.config = config
        self.loss = loss
        self.dataset = dataset
        self.two_layer = two_layer
        self._setup()

    @abstractmethod
    def create_networks(self):
        pass

    def _setup(self):
        encoder, encoder_init, decoder, decoder_init = self.create_networks()
        if self.two_layer:
            elbo = two_layer_objectives.generate_negative_elbo(encoder, decoder)
            vi_loss = TWO_LAYER_LOSSES[self.loss](encoder, decoder)
        else:
            elbo = objectives.generate_negative_elbo(encoder, decoder)
            vi_loss = LOSSES[self.loss](encoder, decoder)

        if self.loss in ['tunable_rebar', 'controlled_score_function']:
            vi_loss, support_fn = vi_loss
        else:
            support_fn = None

        self.training_loop = setup_experiment(
            vi_loss,
            elbo,
            encoder_init,
            decoder_init,
            decoder,
            self.config,
            dataset=self.dataset,
            two_layer=self.two_layer,
            support_fn=support_fn
        )

    def run_and_save(self, prefix=None):
        learning_rate_spec = f'learning_rate_{self.config["learning_rate"]}'
        output_filename = f'{self.loss}_{self.config["image_path"].split("/")[-1]}.json'
        if prefix is not None:
            output_filename = f'{prefix}_{output_filename}'
        output_path = os.path.join(self.output_path, learning_rate_spec)
        states_path = os.path.join(output_path, 'states/')

        if not os.path.exists(states_path):
            os.makedirs(states_path)

        output_file = os.path.join(output_path, output_filename)
        if os.path.exists(output_file):
            pass
        else:
            results, states = self.training_loop(verbose=False)
            with open(output_file, 'w') as f:
                json.dump(results, f)

            states_file = os.path.join(output_path, f'states/states_{output_filename}')
            with open(states_file, 'w') as f:
                json.dump(states, f)


class RelaxInstance(Instance, ABC):
    @abstractmethod
    def create_surrogate(self):
        pass

    def _setup(self):
        encoder, encoder_init, decoder, decoder_init = self.create_networks()
        surrogate, surrogate_init = self.create_surrogate()
        if self.two_layer:
            elbo = two_layer_objectives.generate_negative_elbo(encoder, decoder)
            vi_loss, gradient_variance = TWO_LAYER_LOSSES[self.loss](encoder, decoder, surrogate)
        else:
            elbo = objectives.generate_negative_elbo(encoder, decoder)
            vi_loss, gradient_variance = LOSSES[self.loss](encoder, decoder, surrogate)
        self.training_loop = setup_experiment(
            vi_loss,
            elbo,
            encoder_init,
            decoder_init,
            decoder,
            self.config,
            dataset=self.dataset,
            two_layer=self.two_layer,
            support_fn=gradient_variance,
            surrogate_init=surrogate_init
        )

