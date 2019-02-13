# A big thanks for basic layout of this class to:
# Morgan
# TensorFlow: A proposal of good practices for files, folders and models architecture
# https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
import os
import copy
import json

import numpy as np
import tensorflow as tf

from helpers import check_equal, random_shuffle, load_config
from MFCC import MFCC
from DataLoader import DataLoader
load_cepstra = MFCC.load_cepstra  # for loading  cepstrum-###.npy files from a folder into a list of lists
load_labels = DataLoader.load_labels  # for loading transcript-###.npy files from folder into a list of lists


class AcousticModel(object):

    def __init__(self, config):
        """Initialize the model and it's configuration
        :param config: dictionary (or path to json file) with entries that are to be used for configuring the model
        """

        config = load_config(config)

        # check if config was loaded properly
        if not config:
            raise ValueError("The load_config function helper returned False. The config wasn't loaded properly.")

        # update config with best known configuration
        if config['best']:
            config.update(self.get_best_config(config['env_name']))  # TODO: implement get_best_config and save_config

        # create a deepcopy (nested copy) of the config
        self.config = copy.deepcopy(config)

        # for debugging purposes, prints the loaded config
        if self.config['debug']:
            print('Loaded configuration: ', self.config)

        # SETTINGS #
        self.load_dir = self.config['load_dir']  # directory from which to load MFCC data (works with nested dirs)
        self.save_dir = self.config['save_dir']  # directory in which to save the checkpoints and results
        self.do_train = self.config['do_train']  # if True, training will be commenced, else inference will be commenced
        self.num_cpu_cores = self.config['num_cpu_cores']  # number of CPU cores to use for parallelization

        # Data-inferred parameters (check load_data())#
        self.num_data = None       # total number of individual data in the loaded dataset
        self.max_time = None       # maximal time unrolling of the BiRNN
        self.num_features = None   # number of features in the loaded MFCC cepstra
#        self.x_train = None        # training cepstra [list of numpy arrays] (batch_size, time_length, num_features)
#        self.y_train = None        # training labels [list of numpy arrays] (batch_size, transcript_length)
#        self.x_test = None         # testing cepstra [list of numpy arrays] (batch_size, time_length, num_features)
#        self.y_test = None         # testing labels [list of numpy arrays] (batch_size, transcript_length)
        self.ds_train = None       # tf.Dataset object with elements of training data with components (cepstrum, label)
        self.ds_test = None        # tf.Dataset object with elements of testing data with components (cepstrum, label)
#        self.gen_train = None      # training data generator: yield (cepstrum [nparray], label [nparray])
#        self.gen_test = None       # testing data generator: yield (cepstrum [nparray], label [nparray])
        self.inputs = None          # dictionary with the outputs from batched dataset iterators
        self.outputs = None         # dictionary with outputs from the model

        # HyperParameters (HP) #
        # size of the alphabet in DataLoader
        self.alphabet_size = len(DataLoader.c2n_map)  # number of characters in the alphabet of transcripts
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
            # TODO: make num_hidden into list (length of the list will serve as num_layers)
#            self.num_layers = self.config['num_layers']   # number of stacked layers of LSTM cells in BiRNN
            self.beam_width = self.config['beam_width']   # beam width for the Beam Search (BS) algorithm
            self.top_paths = self.config['top_paths']     # number of best paths to return from the BS algorithm

        # TODO: Set properties/hyperparams function (overriding the 'config' dictionary)

        # TODO: save_configuration()

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        # self.sess = tf.Session(config=sess_config, graph=self.graph)
        # self.sw = tf.summary.FileWriter(self.save_dir, self.sess.graph)

        # load_data()
        self.load_data()
        # prepare data
        self.prepare_data()

        # TODO: build_graph()
        self.graph = self.build_graph()

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        # with self.graph.as_default():
        #     self.saver = tf.train.Saver(
        #         max_to_keep=50,
        #     )

        # TODO: learn_from_epoch()

        # TODO: train(), infer()
        # if self.do_train:
        #    self.train()
        # else:
        #    self.infer()

    def load_data(self):
        """ load cepstra and labels from load_dir, shuffle them and split them into training and testing datasets
        :ivar self.ds_train (tf.data.Dataset object)
        :ivar self.ds_test (tf.data.Dataset object)

        Structure of the elements in instance variables self.ds_train and self.ds_test:
            (cepstrum, label, max_time_cepstrum, length_label)

        :return None

        """
        cepstra = load_cepstra(self.load_dir)
        labels = load_labels(self.load_dir)

        # tests
        assert cepstra[0][0].dtype == np.float64, 'cepstra should be a list of lists with np arrays of dtype float64'
        assert labels[0][0].dtype == np.uint8, 'labels should be a list of lists with np arrays of dtype uint8'

        # flatten the lists to length (sum(n_files for n_files in subfolders))
        cepstra = [item for sublist in cepstra for item in sublist]
        labels = [item for sublist in labels for item in sublist]

        # get the total number of data loaded
        self.num_data = len(cepstra)
        # get number of features in loaded cepstra
        num_features_gen = (cepstrum.shape[1] for cepstrum in cepstra)
        if check_equal(num_features_gen):
            self.num_features = cepstra[0].shape[1]
        else:
            raise ValueError("The number of features is not identical in all loaded cepstrum files.")
        # get maximum number of frames which will serve as a max_time for unrolling the dynamic_rnn
        self.max_time = max(cepstrum.shape[0] for cepstrum in cepstra)

        # shuffle cepstra and labels the same way so that they are still aligned
        cepstra, labels = random_shuffle(cepstra, labels, self.shuffle_seed)

        # split cepstra and labels into traning and testing parts
        len_train = int(self.tt_ratio*self.num_data)  # length of the training data
        len_test = self.num_data - int(self.tt_ratio*self.num_data)  # length of the testing data
        slice_train = slice(0, len_train)  # training part of the data
        slice_test = slice(len_train, None)  # testing part of the data
        x_train = cepstra[slice_train]
        y_train = labels[slice_train]
        x_test = cepstra[slice_test]
        y_test = labels[slice_test]

        # create tf Dataset objects from the training and testing data
        data_types = (tf.float32, tf.int32)
        data_shapes = (tf.TensorShape([None, self.num_features]), tf.TensorShape([None]))

        ds_train = tf.data.Dataset.from_generator(lambda: zip(x_train, y_train),
                                                  data_types,
                                                  data_shapes
                                                  )
        ds_test = tf.data.Dataset.from_generator(lambda: zip(x_test, y_test),
                                                 data_types,
                                                 data_shapes
                                                 )

        # create two more components which contain the sizes of the cepstra and labels
        self.ds_train = ds_train.map(lambda x, y: (x, y, tf.shape(x)[0], tf.size(y)),
                                     num_parallel_calls=self.num_cpu_cores)
        self.ds_test = ds_test.map(lambda x, y: (x, y, tf.shape(x)[0], tf.size(y)),
                                   num_parallel_calls=self.num_cpu_cores)

    def prepare_data(self):
        """Prepare datasets for iteration through the model

        :ivar self.inputs_train
        :ivar self.inputs_test

        :return None

        """

        # combine the elements in datasets into batches of padded components
        padded_shapes = (tf.TensorShape([self.max_time, self.num_features]),  # cepstra padded to self.max_time
                         tf.TensorShape([None]),                              # labels padded to max length in batch
                         tf.TensorShape([]),                                  # sizes not padded
                         tf.TensorShape([]))                                  # sizes not padded
        padding_values = (tf.constant(0.0, dtype=tf.float32),                    # cepstra padded with 0
                          tf.constant(DataLoader.c2n_map[' '], dtype=tf.int32),  # labels padded with blank
                          0,                                                     # size(cepstrum) -- unused
                          0)                                                     # size(label) -- unused

        # TODO: make it work for drop_remainder=False
        ds_train = self.ds_train.padded_batch(self.batch_size, padded_shapes, padding_values,
                                              drop_remainder=True).prefetch(1)
        ds_test = self.ds_test.padded_batch(self.batch_size, padded_shapes, padding_values,
                                            drop_remainder=True).prefetch(1)


        # make initialisable iterator over the dataset which will return the batches of (x, y, size_x, size_y)
        iterator = tf.data.Iterator.from_structure(ds_train.output_types, ds_train.output_shapes)
        x, y, size_x, size_y = iterator.get_next()

        # make initializers over the training and testing datasets
        iterator_train_init = iterator.make_initializer(ds_train)
        iterator_test_init = iterator.make_initializer(ds_test)

        # Build instance dictionary with the iterator data and operations
        self.inputs = {"x": x,
                       "y": y,
                       "size_x": size_x,
                       "size_y": size_y,
                       "init_train": iterator_train_init,
                       "init_test": iterator_test_init}

    @staticmethod
    def lstm_cell(num_hidden):
        # TODO: try to use tf.contrib.cudnn_rnn.CudnnLSTM(num_units=self.num_hidden, state_is_tuple=True)
        # TODO: Batch Normalization at LSTM outputs (before the activation function)
        return tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True, activation='tanh')

    def build_graph(self):
        # TODO: inputs and labels
        # x_placeholder = tf.placeholder(tf.float32, (None, None, self.num_features))
        # y_placeholder = tf.placeholder(tf.int32, (None, None))

        # 1st layer: stacked BiRNN with LSTM cells
        # TODO: consider using tf.contrib.rnn.stack_bidirectional_dynamic_rnn
        cells_fw = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(n) for n in self.num_hidden])  # forward stacked cells
        cells_bw = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(n) for n in self.num_hidden])  # backward stacked cells
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cells_fw,
                                                                  cells_bw,
                                                                  inputs=self.inputs["x"],
                                                                  sequence_length=self.inputs["size_x"],
                                                                  dtype=tf.float32)

        # rnn_outputs == tuple(output_fw, output_bw) ... output_fw == [batch_size, max_time, num_hidden]

        # add the fw and bw outputs together into one tensor
        rnn_outputs = tf.add(rnn_outputs[0], rnn_outputs[1])

        # Reshape output from a tensor of shape [batch_size, max_time, num_hidden]
        # to a tensor of shape [batch_size*max_time, num_hidden]
        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.num_hidden[-1]])

        # 2nd layer: linear projection of outputs from BiRNN
        # define weights and biases for linear projection of outputs from BiRNN
        logit_size = self.alphabet_size + 1  # +1 for the blank
        lp_weights = tf.Variable(tf.random.normal([self.num_hidden[-1], logit_size], dtype=tf.float32))
        lp_biases = tf.Variable(tf.random.normal([logit_size], dtype=tf.float32))

        # convert rnn_outputs into logits (apply linear projection of rnn outputs)
        # lp_outputs.shape == [batch_size*max_time, alphabet_size + 1]
        lp_outputs = tf.nn.relu(tf.add(tf.matmul(rnn_outputs, lp_weights), lp_biases))

        # reshape lp_outputs to shape [batch_size, max_time, alphabet_size + 1]
        logits = tf.reshape(lp_outputs, [self.batch_size, self.max_time, logit_size])

        # convert labels to sparse tensor
        labels = tf.contrib.layers.dense_to_sparse(self.inputs["y"])
        # TODO: check if we need to add eos_token to the transcripts!

        # decode the logits
        # TODO: create separate function for this
        ctc_output = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                   sequence_length=self.inputs["size_x"],
                                                   beam_width=self.beam_width,
                                                   top_paths=self.top_paths,
                                                   merge_repeated=True)
        # !!! ValueError: Dimensions must be equal, but are 13302 and 20 for 'CTCBeamSearchDecoder'
        # (op: 'CTCBeamSearchDecoder') with input shapes: [20,13302,44], [20].
        # !!! inputs: 3-D float Tensor, size [max_time, batch_size, num_classes]. The logits.
        # TODO: switch the max_time and batch_size dimensions

        # calculate ctc loss of logits
        ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=self.inputs["size_x"])
        # !!!TypeError: Expected labels (first argument) to be a SparseTensor

        self.outputs = {"ctc_output": ctc_output,
                        "ctc_loss": ctc_loss,
                        }


        # tf.nn.batch_normalization()
        # TODO: add rnn.DropoutWrapper
        # TODO: FC layers at every timestep output W(num_classes, num_hidden) -> (num_classes, 1) at every timestep
        # TODO: ReLU at every FC output
        # TODO: ctc_beam_search_decoder -> outputs (only for validation and inference)
        # TODO: tf.reduce_mean(ctc_loss) -> cost
        # TODO: optimizer
        # TODO: minimize(cost)
        # raise Exception('The build_graph function must be implemented')

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



