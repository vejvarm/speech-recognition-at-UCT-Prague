# A big thanks for basic layout of this class to:
# Morgan
# TensorFlow: A proposal of good practices for files, folders and models architecture
# https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
import os
import copy
import json

import numpy as np
import tensorflow as tf

from helpers import check_equal, random_shuffle
from MFCC import MFCC
from DataLoader import DataLoader
load_cepstra = MFCC.load_cepstra  # for loading  cepstrum-###.npy files from a folder into a list of lists
load_labels = DataLoader.load_labels  # for loading transcript-###.npy files from folder into a list of lists

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
        self.do_train = self.config['do_train']  # if True, training will be commenced, else inference will be commenced

        # Data-inferred parameters (check load_data())#
        self.num_data = None      # total number of individual data in the loaded dataset
        self.max_time = None      # maximal time unrolling of the BiRNN
        self.num_features = None  # number of features in the loaded MFCC cepstra

        # HyperParameters (HP) #
        if config['random']:  # TODO: The missing (None) params will be configured randomly
            pass  # TODO: random_config() for generating random configurations of the hyperparams
        else:
            # training HP
            self.lr = self.config['lr']                      # learning rate
            self.max_epochs = self.config['max_epochs']      # maximum number of training epochs
            self.batch_size = self.config['batch_size']      # size of mini_batches to be fed into the net at once
            self.tt_ratio = self.config['tt_ratio']          # train-test data split ratio
            self.shuffle_seed = self.config['shuffle_seed']  # seed for shuffling the cepstra and labels

            # AcousticModel specific HP
            self.num_hidden = self.config['num_hidden']   # number of hidden units in LSTM cells
            self.num_layers = self.config['num_layers']   # number of stacked layers of LSTM cells in BiRNN
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
        if self.do_train:
            self.train()
        else:
            self.infer()

    def load_data(self):
        """load cepstra and transcripts from load_dir and split them into training and testing sets randomly"""
        cepstra = load_cepstra(self.load_dir)
        labels = load_labels(self.load_dir)

        # tests
        assert cepstra[0][0].dtype == np.float64, 'cepstra should be a list of lists with np arrays of dtype float64'
        assert labels[0][0].dtype == np.uint8, 'labels should be a list of lists with np arrays of dtype uint8'

        # flatten the lists into tuples of length (sum(n_files for n_files in subfolders))
        flat_cepstra = tuple(item for sublist in cepstra for item in sublist)
        flat_labels = tuple(item for sublist in labels for item in sublist)

        # get the total number of data loaded
        self.num_data = len(flat_cepstra)
        # get number of features in loaded cepstra
        num_features_gen = (cepstrum.shape[1] for cepstrum in flat_cepstra)
        if check_equal(num_features_gen):
            self.num_features = flat_cepstra[0].shape[1]
        else:
            raise ValueError("The number of features is not identical in all loaded cepstrum files.")
        # get maximum number of frames which will serve as a max_time for unrolling the dynamic_rnn
        self.max_time = max(cepstrum.shape[0] for cepstrum in flat_cepstra)

        # shuffle cepstra and labels the same way so that they are still aligned
        shuffled_cepstra, shuffled_labels = random_shuffle(flat_cepstra, flat_labels, self.shuffle_seed)

        # convert cepstra and labels into tuples of tensors
        tf_cepstra = tuple(tf.constant(array) for array in shuffled_cepstra)
        tf_labels = tuple(tf.constant(array) for array in shuffled_labels)

        # split cepstra and labels into traning and testing parts
        len_train = int(self.tt_ratio*self.num_data)                 # length of the training data
        len_test = self.num_data - int(self.tt_ratio*self.num_data)  # length of the testing data
        slice_train = slice(0, len_train)                            # training part of the data
        slice_test = slice(len_train+1, None)                        # testing part of the data
        x_train = tf_cepstra[slice_train]
        y_train = tf_labels[slice_train]
        x_test = tf_labels[slice_test]
        y_test = tf_labels[slice_test]

        # create tf.data.Dataset objects from training data
        x_train = tf.data.Dataset.from_tensors(x_train)
        y_train = tf.data.Dataset.from_tensors(y_train)
        x_test = tf.data.Dataset.from_tensors(x_test)
        y_test = tf.data.Dataset.from_tensors(y_test)

        # create tf.data.Dataset objects of max_time-padded batches of training and testing data
        pad_shape_train = tuple([[self.max_time, self.num_features]]*len_train)
        pad_shape_test = tuple([[self.max_time, self.num_features]]*len_test)
        x_train_batches = x_train.padded_batch(self.batch_size, padded_shapes=pad_shape_train)
        y_train_batches = y_train.padded_batch(self.batch_size, padded_shapes=pad_shape_train)
        x_test_batches = x_test.padded_batch(self.batch_size, padded_shapes=pad_shape_test)
        y_test_batches = y_test.padded_batch(self.batch_size, padded_shapes=pad_shape_test)

        # TODO: Create (reinitializable) iterators from the Datasets of max_time-padded batches

        # for extracting a batch, just call e.g. x_train_batches.


    def load_batch(self):
        pass


    def lstm_cell(self):
        return tf.nn.rnn_cell.LSTMCell(num_units=self.num_hidden, state_is_tuple=True)

    def build_graph(self):
        # TODO: inputs and labels
        x_train = tf.placeholder(tf.float32, [None, self.max_time, self.num_features])
        x_test = tf.placeholder()

        # TODO: BiRNN with LSTM cells
        cells_fw = [self.lstm_cell() for _ in range(self.num_layers)]  # forward direction LSTM cells
        cells_bw = [self.lstm_cell() for _ in range(self.num_layers)]  # backward direction LSTM cells
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cells_fw, cells_bw, )

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



