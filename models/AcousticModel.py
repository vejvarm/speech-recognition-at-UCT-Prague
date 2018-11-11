import os
import copy
import json

import numpy as np
import tensorflow as tf


class AcousticModel(object):

    def __init__(self, config):
        """Initialize the model and it's configuration
        :param config: dictionary (from json file) with entries that are to be used for configuring the model
        """

        # update config with best known configuration
        if config['best']:
            config.update(self.get_best_config(config['env_name']))  # TODO: implement get_best_config and save_config

        # create a deepcopy (nested copy) of the config
        self.config = copy.deepcopy(config)

        # for debugging purposes, prints the loaded config
        if self.config['debug']:
            print('Loaded configuration: ', self.config)

        # SETTINGS #
        self.load_dir = self.config['load_dir']  # directory from which to load data TODO: can be nested with other dirs
        self.save_dir = self.config['save_dir']  # directory in which to save the checkpoints and results
        self.train = self.config['train']        # if True, training will be commenced, else inference will be commenced

        # HyperParameters (HP) #
        if config['random']:  # TODO: The missing (None) params will be configured randomly
            pass  # TODO: random_config() for generating random configurations of the hyperparams
        else:
            # training HP
            self.lr = self.config['lr']                   # learning rate
            self.max_epochs = self.config['max_epochs']   # maximum number of training epochs

            # AcousticModel specific HP
            self.num_hidden = self.config['num_hidden']   # number of hidden units in LSTM cells
            self.num_layers = self.config['num_layers']   # number of stacked layers of LSTM cells in BiRNN
            self.max_time = self.config['max_time']       # maximal time unrolling of the BiRNN
            self.beam_width = self.config['beam_width']   # beam width for the Beam Search (BS) algorithm
            self.top_paths = self.top_paths['top_paths']  # number of best paths to return from the BS algorithm

        # TODO: Set properties/hyperparams function (overriding the 'config' dictionary)

        # TODO: save_configuration()

        # TODO: build_graph()
        self.graph = self.build_graph(tf.Graph())

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=50,
            )

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.save_dir, self.sess.graph)

        # TODO: load_data()

        # TODO: learn_from_epoch()

        # TODO: train(), infer()

    def build_graph(self):
        # TODO: BiRNN with LSTM cells
        # TODO: FC layers at every timestep output W(num_classes, num_hidden) -> (num_classes, 1) at every timestep
        # TODO: ReLU at every FC output
        # TODO: ctc_beam_search_decoder -> outputs (only for validation and inference)
        # TODO: tf.reduce_mean(ctc_loss) -> cost
        # TODO: optimizer
        # TODO: minimize(cost)
        raise Exception('The build_graph function must be implemented')

    def infer(self):
        raise Exception('The infer function must be implemented')

    def learn_from_epoch(self):
        raise Exception('The learn_from_epoch function must be implemented')

    def train(self, save_every=1):
        # This function is usually common to all your models, Here is an example:
        for epoch_id in range(0, self.max_epochs):
            self.learn_from_epoch()

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and epoch_id % save_every == 0:
                self.save()

    def save(self):
        # This function is usually common to all your models, Here is an example:
        global_step_t = tf.train.get_global_step(self.graph)
        global_step, episode_id = self.sess.run([global_step_t, self.episode_id])
        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.save_dir, global_step))
        self.saver.save(self.sess, self.save_dir + '/agent-ep_' + str(episode_id), global_step)

        # I always keep the configuration that
        if not os.path.isfile(self.save_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.save_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)



