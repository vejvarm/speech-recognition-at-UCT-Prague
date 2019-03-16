# A big thanks for basic layout of this class to:
# Morgan
# TensorFlow: A proposal of good practices for files, folders and models architecture
# https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
import copy
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow.contrib import rnn, cudnn_rnn

from DataLoader import DataLoader
from MFCC import MFCC
from helpers import check_equal, load_config, get_available_devices, load_speech, list_to_padded_array

load_cepstra = MFCC.load_cepstra  # for loading  cepstrum-###.npy files from a folder into a list of lists
load_labels = DataLoader.load_labels  # for loading transcript-###.npy files from folder into a list of lists


# TODO: try using tf.contrib.rnn.GridLSTMCell instead of stacked bidirectional LSTM
class AcousticModel(object):
    load_dir: str
    save_dir: str
    do_train: bool
    from_checkpoint: bool
    checkpoint_dir: str
    num_cpu_cores: int
    parallel_iterations: int
    cepstrum_pad_val: float
    label_pad_val: int
    num_data: int
    max_time: int
    num_features: int
    ds_train: tf.data.Dataset
    ds_test: tf.data.Dataset
    inputs: Dict[str, Union[tf.Tensor, tf.Operation]]
    outputs: Dict[str, Union[tf.Tensor, tf.Operation]]
    global_step: tf.Tensor
    increment_global_step_op: tf.Operation
    lr: float
    max_epochs: int
    batch_size: int
    tt_ratio: float
    shuffle_seed: int
    decay_by_epochs: bool
    decay_by_length: bool
    d_by_epo_rate: float
    d_by_epo_steps: int
    d_by_len_rate: float
    d_by_len_steps: int
    ff_num_hidden: List[int]
    ff_dropout: List[float]
    ff_batch_norm: bool
    relu_clip: float
    rnn_num_hidden: List[int]
    rnn_use_peephole: bool
    ctc_collapse_repeated: bool
    ctc_merge_repeated: bool
    beam_width: int
    top_paths: int
    grad_clip: Union[str, bool]
    grad_clip_val: float
    show_device_placement: bool
    print_batch_x: bool
    print_layer_3: bool
    print_dropout: bool
    print_rnn_outputs: bool
    print_lr: bool
    print_gradients: bool
    print_grad_norm: bool
    print_labels: bool
    print_batch_x_op: tf.print
    print_layer_3_op: tf.print
    print_dropout_op: tf.print
    print_gradients_op: tf.print
    print_grad_norm_op: tf.print
    print_labels_op: tf.print
    episode_id: str

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
        self.from_checkpoint = self.config['from_checkpoint']  # if True, loads last checkpoint in save_dir
        self.checkpoint_dir = self.config['checkpoint_dir']  # directory from which to load checkpoint
        self.num_cpu_cores = self.config['num_cpu_cores']  # number of CPU cores to use for parallelization
        self.parallel_iterations = self.config['parallel_iterations']  # GPU parallelization in stacked_dynamic_BiRNN
        self.cepstrum_pad_val = self.config['cepstrum_pad_val']  # value with which to pad cepstra to same length
        self.label_pad_val = self.config['label_pad_val']  # value to pad batches of labels to same length
        self.mfcc_deltas = self.config['mfcc_deltas']  # which deltas were set during MFCC generation for training
        self.init_op = None

        # Data-inferred parameters (check load_data())#
        self.num_data = None  # total number of individual data in the loaded dataset
        self.max_time = None  # maximal time unrolling of the BiRNN
        self.num_features = None  # number of features in the loaded MFCC cepstra
        self.ds_train = None  # tf.Dataset object with elements of training data with components (cepstrum, label)
        self.ds_test = None  # tf.Dataset object with elements of testing data with components (cepstrum, label)
        self.inputs = None  # dictionary with the inputs to model from batched dataset iterators
        self.outputs = None  # dictionary with outputs from the model
        self.global_step = None  # (int) counter for current epoch
        self.batch_no = None  # (int) counter for current batch number
        self.increment_global_step_op = None  # operation for incrementing global_step by 1
        self.increment_batch_no_op = None  # operation for incrementing batch_no by 1
        self.total_loss = None  # (float) total loss in one epoch
        self.epoch_mean_cer = None  # (float) mean characer error rate (CER) throughout the epoch
        self.increment_total_loss = None  # operation for incrementing total_loss by mean_loss
        self.update_epoch_mean_cer = None  # opearation for upgrading epoch_mean_cer by mean_cer
        self.reset_total_loss = None  # operation for resetting total_loss back to 0 (at start of each epoch)
        self.reset_batch_no = None  # operation for resetting the batch number

        # HyperParameters (HP) #
        # size of the alphabet in DataLoader
        self.alphabet_size = len(DataLoader.c2n_map)  # number of characters in the alphabet of transcripts
        if config['random']:  # TODO: The missing (None) params will be configured randomly
            pass  # TODO: random_config() for generating random configurations of the hyperparams
        else:
            # training HP
            self.lr = self.config['lr']  # (float) learning rate
            self.max_epochs = self.config['max_epochs']  # (int) maximum number of training epochs
            self.batch_size = self.config['batch_size']  # (int) size of mini-batches during learning from epoch
            self.tt_ratio = self.config['tt_ratio']  # (float) train-test data split ratio
            self.shuffle_seed = self.config['shuffle_seed']  # (int) seed for shuffling the cepstra and labels

            # decay
            self.decay_by_epochs = self.config['decay_by_epochs']
            self.decay_by_length = self.config['decay_by_length']
            self.d_by_epo_rate = self.config['d_by_epo_rate']
            self.d_by_epo_steps = self.config['d_by_epo_steps']
            self.d_by_len_rate = self.config['d_by_len_rate']
            self.d_by_len_steps = self.config['d_by_len_steps']

            # AcousticModel specific HP
            self.ff_num_hidden = self.config['ff_num_hidden']  # (list of ints) hidden units in feed forward layers
            self.ff_dropout = self.config['ff_dropout']  # (list of floats) dropouts after each feed forward layer
            self.ff_batch_norm = self.config['ff_batch_norm']  # (bool) use batch normalisation in feed forward layers
            self.relu_clip = self.config['relu_clip']  # (float) preventing exploding gradient with relu clipping
            self.rnn_num_hidden = self.config['rnn_num_hidden']  # (list of ints) number of hidden units in LSTM cells
            self.rnn_use_peephole = self.config['rnn_use_peephole']  # (bool) use peephole connections in the LSTM cells
            self.ctc_collapse_repeated = self.config['ctc_collapse_repeated']  # (bool)
            self.ctc_merge_repeated = self.config['ctc_merge_repeated']  # (bool)
            self.beam_width = self.config['beam_width']  # (int) beam width for the Beam Search (BS) algorithm
            self.top_paths = self.config['top_paths']  # (int) number of best paths to return from the BS algorithm
            self.grad_clip = self.config['grad_clip']  # (str|bool) method to be used (False for no clipping)
            self.grad_clip_val = self.config['grad_clip_val']  # (float) value at which to clip gradient

        # DEBUGGING SETTINGS # (works only if config["debug"] == True)
        self.show_device_placement = self.config['show_device_placement']
        self.print_batch_x = self.config['print_batch_x']
        self.print_layer_3 = self.config['print_layer_3']
        self.print_dropout = self.config['print_dropout']
        self.print_rnn_outputs = self.config['print_rnn_outputs']
        self.print_lr = self.config['print_lr']
        self.print_gradients = self.config['print_gradients']  # (bool) print out gradients at each batch
        self.print_grad_norm = self.config['print_grad_norm']  # (bool) print out gradient norm at each batch
        self.print_labels = self.config['print_labels']

        # SUMMARIES #
        self.merged_summaries = None

        # PRINT OPERATIONS #
        self.print_batch_x_op = None
        self.print_layer_3_op = None
        self.print_dropout_op = None
        self.print_rnn_outputs_op = None
        self.print_lr_op = None
        self.print_gradients_op = None
        self.print_grad_norm_op = None
        self.print_labels_op = None

        self.episode_id = datetime.now().strftime('%Y-%m%d-%H%M%S')  # unique episode id from current date and time
        self.checkpoint_save_path = self.save_dir + '/agent-ep_' + str(self.episode_id)

        # TODO: Set properties/hyperparams function (overriding the 'config' dictionary)

        # TODO: save_configuration()

        self.graph = tf.Graph()

        # load_data()
        self.load_data()
        # prepare data
        self.prepare_data()

        self.build_graph()

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=50,
            )

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        logging_options = True if self.config["debug"] and self.show_device_placement else False
        sess_config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=logging_options)
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        self.sw_train = tf.summary.FileWriter(self.checkpoint_save_path + '/train', self.sess.graph)
        self.sw_test = tf.summary.FileWriter(self.checkpoint_save_path + '/test')

        # TODO: train(), infer()
        # if self.do_train:
        #    self.train()
        # else:
        #    self.infer()
        self.init()

    def load_data(self):
        """ load cepstra and labels from load_dir, shuffle them and split them into training and testing datasets
        :ivar self.ds_train (tf.data.Dataset object)
        :ivar self.ds_test (tf.data.Dataset object)

        Structure of the elements in instance variables self.ds_train and self.ds_test:
            (cepstrum, label, max_time_cepstrum, length_label)

        :return None

        """
        cepstra, _ = load_cepstra(self.load_dir)
        labels, _ = load_labels(self.load_dir)

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
        # get number of time frames of the longest cepstrum
        self.max_time = max(cepstrum.shape[0] for cepstrum in cepstra)

        # shuffle cepstra and labels the same way so that they are still aligned
        # !!! TODO: uncomment or make shuffle part of dataset pipeline!
        # cepstra, labels = random_shuffle(cepstra, labels, self.shuffle_seed)

        # split cepstra and labels into traning and testing parts
        len_train = int(self.tt_ratio * self.num_data)  # length of the training data
        len_test = self.num_data - int(self.tt_ratio * self.num_data)  # length of the testing data
        slice_train = slice(0, len_train)  # training part of the data
        slice_test = slice(len_train, None)  # testing part of the data
        x_train = cepstra[slice_train]
        y_train = labels[slice_train]
        x_test = cepstra[slice_test]
        y_test = labels[slice_test]

        with self.graph.as_default():
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

        num_train_batches = int(self.num_data * self.tt_ratio // self.batch_size)
        num_test_batches = int(self.num_data * (1 - self.tt_ratio) // self.batch_size)

        with self.graph.as_default():
            # combine the elements in datasets into batches of padded components
            padded_shapes = (tf.TensorShape([None, self.num_features]),  # cepstra padded to maximum time in batch
                             tf.TensorShape([None]),  # labels padded to maximum length in batch
                             tf.TensorShape([]),  # sizes not padded
                             tf.TensorShape([]))  # sizes not padded
            padding_values = (tf.constant(self.cepstrum_pad_val, dtype=tf.float32),  # cepstra padded with 0.0
                              tf.constant(self.label_pad_val, dtype=tf.int32),  # labels padded with -1
                              0,  # size(cepstrum) -- unused
                              0)  # size(label) -- unused

            # TODO: make it work for drop_remainder=False
            ds_train = self.ds_train.padded_batch(self.batch_size, padded_shapes, padding_values,
                                                  drop_remainder=True)
            ds_test = self.ds_test.padded_batch(self.batch_size, padded_shapes, padding_values,
                                                drop_remainder=True)

            # shuffle the batches (simillar length stays in one batch but the order of batches is shuffled every epoch
            ds_train = ds_train.shuffle(buffer_size=num_train_batches,
                                        seed=self.shuffle_seed,
                                        reshuffle_each_iteration=True).prefetch(1)
            ds_test = ds_test.shuffle(buffer_size=num_test_batches,
                                      seed=self.shuffle_seed,
                                      reshuffle_each_iteration=True).prefetch(1)

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
    def ff_layer(w_name, b_name, input_size, output_size):
        w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.001), name=w_name)
        b = tf.Variable(tf.truncated_normal([output_size], stddev=0.001), name=b_name)
        return w, b

    @staticmethod
    def batch_norm_layer(x, is_train, scope):
        # !!! during training the tf.GraphKeys.UPDATE_OPS must be called to update the mean and variance
        bn = tf.contrib.layers.batch_norm(x,
                                          decay=0.9,
                                          is_training=is_train,
                                          updates_collections=tf.GraphKeys.UPDATE_OPS,
                                          zero_debias_moving_mean=True,
                                          scope=scope)
        return bn

    def lstm_cell(self, num_hidden):
        # TODO: try to use tf.contrib.cudnn_rnn.CudnnLSTM(num_units=self.rnn_num_hidden, state_is_tuple=True)
        # TODO: Batch Normalization at LSTM outputs (before the activation function)
        return rnn.LSTMBlockCell(num_units=num_hidden, use_peephole=self.rnn_use_peephole)
        # cell = rnn.LSTMBlockFusedCell(num_units=num_hidden)
        # return cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden, use_peephole=self.rnn_use_peephole)
        # return tf.contrib.grid_rnn.Grid1LSTMCell(num_units=num_hidden, state_is_tuple=True)
        # return rnn.GridLSTMCell(num_units=num_hidden, state_is_tuple=True, num_frequency_blocks=None)
        # return tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True, activation='tanh')

    def decaying_learning_rate(self):
        """ reduce learning rate based on number of epochs and/or length of the sequence

        :return: decayed learning rate
        """
        with self.graph.as_default():
            lr = tf.constant(self.lr, dtype=tf.float32)
            if self.decay_by_epochs:
                lr = tf.train.exponential_decay(lr, self.global_step, self.d_by_epo_steps, self.d_by_epo_rate)
            if self.decay_by_length:
                mean_seq_len = tf.reduce_mean(self.inputs["size_x"])
                lr = tf.train.exponential_decay(lr, mean_seq_len, self.d_by_len_steps, self.d_by_len_rate)
            return lr

    def build_graph(self):
        # TODO: inputs and labels

        devices = get_available_devices()

        with self.graph.as_default():

            # placeholders
            ph_x = tf.placeholder_with_default(self.inputs["x"], (None, None, self.num_features), name="ph_x")
            ph_size_x = tf.placeholder_with_default(self.inputs["size_x"], (None), name="ph_size_x")
            ph_y = tf.placeholder_with_default(self.inputs["y"], (None, None), name="ph_y")
            ph_batch_size = tf.placeholder(tf.int32, name="ph_batch_size")
            ph_is_train = tf.placeholder(tf.bool, name="ph_is_train")
            ph_ff_dropout = tf.placeholder(tf.float32, [len(self.ff_dropout)], name="ph_ff_dropout")

            # Step tracking tensors and update ops
            self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
            self.batch_no = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
            self.increment_global_step_op = tf.assign_add(self.global_step, 1)
            self.increment_batch_no_op = tf.assign_add(self.batch_no, 1)
            self.reset_batch_no = tf.assign(self.batch_no, 0)

            # SUMMARY tensors
            self.total_loss = tf.Variable(0, trainable=False, name='total_loss', dtype=tf.float32)
            self.epoch_mean_cer = tf.Variable(0, trainable=False, name='epoch_mean_cer', dtype=tf.float32)

            # reshaping from [batch_size, batch_time, num_features] to [batch_size*batch_time, num_features]
            with tf.device(devices["cpu"][0]):
                batch_x = tf.reshape(ph_x, [-1, self.num_features])  # reshape to [batch_size*batch_time, num_features]

                # feed forward layers
                # 1st layer
                with tf.variable_scope("layer_1") as scope:
                    with tf.name_scope("fc_1"):
                        w1, b1 = self.ff_layer("w1", "b1", self.num_features, self.ff_num_hidden[0])
                        layer_1 = tf.add(tf.matmul(batch_x, w1), b1)
                        if self.ff_batch_norm:
                            layer_1 = tf.reshape(layer_1, [ph_batch_size, -1, self.ff_num_hidden[0]])
                            layer_1 = self.batch_norm_layer(layer_1, ph_is_train, scope)
                            layer_1 = tf.reshape(layer_1, [-1, self.ff_num_hidden[0]])
                        # layer_1 = tf.minimum(tf.nn.relu(layer_1), self.relu_clip)
                        # layer_1 = tf.tanh(layer_1)
                        layer_1 = tf.minimum(tf.nn.elu(layer_1), self.relu_clip)
                        layer_1 = tf.nn.dropout(layer_1, keep_prob=(1.0 - ph_ff_dropout[0]))

                # 2nd layer
                with tf.variable_scope("layer_2") as scope:
                    with tf.name_scope("fc_2"):
                        w2, b2 = self.ff_layer("w2", "b2", self.ff_num_hidden[0], self.ff_num_hidden[1])
                        layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
                        if self.ff_batch_norm:
                            layer_2 = tf.reshape(layer_2, [ph_batch_size, -1, self.ff_num_hidden[1]])
                            layer_2 = self.batch_norm_layer(layer_2, ph_is_train, scope)
                            layer_2 = tf.reshape(layer_2, [-1, self.ff_num_hidden[1]])
                        # layer_2 = tf.minimum(tf.nn.relu(layer_2), self.relu_clip)
                        # layer_2 = tf.tanh(layer_2)
                        layer_2 = tf.minimum(tf.nn.elu(layer_2), self.relu_clip)
                        layer_2 = tf.nn.dropout(layer_2, keep_prob=(1.0 - ph_ff_dropout[1]))

                # 3rd layer
                with tf.variable_scope("layer_3") as scope:
                    with tf.name_scope("fc_3"):
                        w3, b3 = self.ff_layer("w3", "b3", self.ff_num_hidden[1], self.ff_num_hidden[2])
                        layer_3 = tf.add(tf.matmul(layer_2, w3), b3)
                        if self.ff_batch_norm:
                            layer_3 = tf.reshape(layer_3, [ph_batch_size, -1, self.ff_num_hidden[2]])
                            layer_3 = self.batch_norm_layer(layer_3, ph_is_train, scope)
                            layer_3 = tf.reshape(layer_3, [-1, self.ff_num_hidden[2]])
                        # layer_3 = tf.minimum(tf.nn.relu(layer_3), self.relu_clip)
                        # layer_3 = tf.tanh(layer_3)
                        layer_3 = tf.minimum(tf.nn.elu(layer_3), self.relu_clip)
                        layer_3 = tf.nn.dropout(layer_3, keep_prob=(1.0 - ph_ff_dropout[2]))
                        # reshape back to [batch_size, batch_time, ff_num_hidden[2]]
                        layer_3 = tf.reshape(layer_3, [ph_batch_size, -1, self.ff_num_hidden[2]])
                        # transpose into time major tensor [batch_time, batch_size, ff_num_hidden[2]] for the rnn input
                        layer_3 = tf.transpose(layer_3, [1, 0, 2])

            # 4th layer: stacked BiRNN with LSTM cells
            # TODO: LSTM might become a computational bottleneck
            with tf.variable_scope("layer_4") as scope:
                with tf.name_scope("birnn"):
                    cells_fw = [self.lstm_cell(n) for n in self.rnn_num_hidden]  # list of forward direction cells
                    cells_bw = [self.lstm_cell(n) for n in self.rnn_num_hidden]  # list of backward direction cells
#                    initial_states_fw = [rnn.LSTMStateTuple(tf.truncated_normal([ph_batch_size, n], stddev=0.00001),
#                                         tf.truncated_normal([ph_batch_size, n], stddev=0.001))
#                                         for n in self.rnn_num_hidden]
#                    initial_states_bw = [rnn.LSTMStateTuple(tf.truncated_normal([ph_batch_size, n], stddev=0.00001),
#                                         tf.truncated_normal([ph_batch_size, n], stddev=0.001))
#                                         for n in self.rnn_num_hidden]
                    rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                            cells_bw,
                                                                            inputs=layer_3,
#                                                                            initial_states_fw=initial_states_fw,
#                                                                            initial_states_bw=initial_states_bw,
                                                                            dtype=tf.float32,
                                                                            sequence_length=ph_size_x,
                                                                            parallel_iterations=self.parallel_iterations,
                                                                            time_major=True)
                    # rnn_outputs: Tensor of shape [batch_time, batch_size, 2*num_hidden]

                    # apply clipped ELU activation to the rnn outputs
                    rnn_outputs = tf.minimum(tf.nn.elu(rnn_outputs), self.relu_clip)

                    # Reshape output from a tensor of shape [batch_time, batch_size, 2*num_hidden]
                    # to a tensor of shape [batch_time*batch_size, 2*num_hidden]
                    rnn_outputs = tf.reshape(rnn_outputs, [-1, 2*self.rnn_num_hidden[-1]])

            # 5th layer: linear projection of outputs from BiRNN
            with tf.name_scope("fc_logits"):
                # define weights and biases for linear projection of outputs from BiRNN
                logit_size = self.alphabet_size + 1  # +1 for the blank
                w5, b5 = self.ff_layer("w5", "b5", 2*self.rnn_num_hidden[-1], logit_size)

                # convert rnn_outputs into logits (apply linear projection of rnn outputs)
                # lp_outputs.shape == [batch_time*batch_size, alphabet_size + 1]
                lp_outputs = tf.minimum(tf.nn.elu(tf.add(tf.matmul(rnn_outputs, w5), b5)), self.relu_clip)

                # reshape lp_outputs to shape [batch_time, batch_size, alphabet_size + 1]
                logits = tf.reshape(lp_outputs, [-1, ph_batch_size, logit_size])

            # convert labels to sparse tensor
            with tf.name_scope("labels"):
                labels = tf.contrib.layers.dense_to_sparse(ph_y, eos_token=self.label_pad_val)

            # calculate ctc loss of logits
            with tf.name_scope("ctc_loss"):
                ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=logits,
                                          sequence_length=ph_size_x,
                                          preprocess_collapse_repeated=self.ctc_collapse_repeated,
                                          ctc_merge_repeated=self.ctc_merge_repeated)

                # Calculate the size normalized average loss across the batch
                mean_loss = tf.reduce_mean(tf.divide(ctc_loss, tf.cast(ph_size_x, tf.float32)))

                self.increment_total_loss = tf.assign_add(self.total_loss, mean_loss)
                self.reset_total_loss = tf.assign(self.total_loss, 0)

            # operation for updating variables in batch normalisation layers during training
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # learning rate decay
            with tf.name_scope("train"):
                lr = self.decaying_learning_rate()

                # use AdamOptimizer to compute the gradients and minimize the average of ctc_loss (training the model)
                # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                # optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
                # optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
                optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

                gradients, variables = zip(*optimizer.compute_gradients(mean_loss))
                # Gradient clipping
                if self.grad_clip == "global_norm":
                    gradients, global_norm = tf.clip_by_global_norm(gradients, self.grad_clip_val)
                elif self.grad_clip == "local_norm":
                    gradients = [None if gradient is None else tf.clip_by_norm(gradient, self.grad_clip_val)
                                 for gradient in gradients]
                    global_norm = tf.global_norm(gradients)
                else:
                    global_norm = tf.global_norm(gradients)
                # update_ops needs to be called for proper update of the trainable variables in batch normalization layers
                with tf.control_dependencies(update_ops):
                    optimize = optimizer.apply_gradients(zip(gradients, variables))

            # decode the logits
            with tf.name_scope("outputs"):
                ctc_outputs, ctc_log_probs = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                                           sequence_length=ph_size_x,
                                                                           beam_width=self.beam_width,
                                                                           top_paths=self.top_paths,
                                                                           merge_repeated=self.ctc_merge_repeated)

                # cast ctc_outputs to int32
                ctc_outputs = [tf.cast(output, tf.int32) for output in ctc_outputs]

                # convert outputs from sparse to dense
                dense_outputs = [tf.sparse.to_dense(output, default_value=self.label_pad_val) for output in ctc_outputs]

            # calculate mean of character error rate (CER) using Levenshtein distance
            with tf.name_scope("error_rate"):
                mean_cer = tf.reduce_mean(tf.edit_distance(ctc_outputs[-1], labels, name="levenshtein_distance"))

                bn_float = tf.cast(self.batch_no, tf.float32)
                current_mean_cer = tf.divide(tf.add(tf.multiply(self.epoch_mean_cer,
                                                                tf.subtract(bn_float, 1.0)),
                                                    mean_cer),
                                             bn_float)

                self.update_epoch_mean_cer = tf.assign(self.epoch_mean_cer, current_mean_cer)

            # add the tensors and ops to a collective dictionary
            self.outputs = {"ctc_outputs": dense_outputs,
                            "ctc_log_probs": ctc_log_probs,
                            "ctc_loss": ctc_loss,
                            "mean_loss": mean_loss,
                            "mean_cer": mean_cer,
                            "optimize": optimize}

            # initializer for TensorFlow variables
            self.init_op = [tf.global_variables_initializer(),
                            tf.local_variables_initializer()]

            # SUMMARIES
            with tf.name_scope("summaries"):
                tf.summary.scalar('total_loss', self.total_loss)
                tf.summary.scalar('mean_cer', self.epoch_mean_cer)

                self.merged_summaries = tf.summary.merge_all()

            # PRINT operations:
            with tf.name_scope("prints"):
                if self.print_batch_x:
                    self.print_batch_x_op = tf.print("batch_x: ", batch_x, "shape: ", batch_x.shape,
                                                     output_stream=sys.stdout)
                if self.print_layer_3:
                    self.print_layer_3_op = tf.print("layer_3: ", layer_3, "shape: ", layer_3.shape,
                                                     output_stream=sys.stdout)
                if self.print_dropout:
                    self.print_dropout_op = tf.print("dropout: ", ph_ff_dropout,
                                                     "shape: ", ph_ff_dropout.shape,
                                                     output_stream=sys.stdout)
                if self.print_rnn_outputs:
                    self.print_rnn_outputs_op = tf.print("rnn_outputs: ", rnn_outputs,
                                                         "shape: ", rnn_outputs.shape,
                                                         output_stream=sys.stdout)
                if self.print_lr:
                    self.print_lr_op = tf.print("learning rate: ", lr,
                                                output_stream=sys.stdout)
                if self.print_gradients:
                    self.print_gradients_op = tf.print("gradients: ", gradients,
                                                       "name: ", [variable.name for variable in variables],
                                                       "shape: ", [gradient.shape for gradient in gradients],
                                                       "maximum: ", ([tf.reduce_max(gradient) for gradient in gradients]),
                                                       output_stream=sys.stdout)
                if self.print_grad_norm:
                    self.print_grad_norm_op = tf.print("Grad. global norm:", global_norm,
                                                       output_stream=sys.stdout)
                if self.print_labels:
                    self.print_labels_op = tf.print("labels: ", ph_y,
                                                    "shape: ", ph_y.shape,
                                                    output_stream=sys.stdout)

    def learn_from_epoch(self):
        output = None

        num_train_batches = int(self.num_data * self.tt_ratio // self.batch_size)
        num_test_batches = int(self.num_data * (1 - self.tt_ratio) // self.batch_size)

        epoch = self.sess.run(self.global_step)

        print("\n_____EPOCH %d" % epoch)
        # TRAINING Dataset
        self.sess.run(self.inputs["init_train"])
        print("\n_____TRAINING DATA_____")
        train_tensors = [self.outputs["mean_loss"],
                         self.outputs["ctc_outputs"],
                         self.outputs["mean_cer"],
                         self.increment_total_loss,
                         self.update_epoch_mean_cer,
                         self.outputs["optimize"]]
        if self.config["debug"]:
            if self.print_batch_x:
                train_tensors.append(self.print_batch_x_op)
            if self.print_layer_3:
                train_tensors.append(self.print_layer_3_op)
            if self.print_dropout:
                train_tensors.append(self.print_dropout_op)
            if self.print_rnn_outputs:
                train_tensors.append(self.print_rnn_outputs_op)
            if self.print_lr:
                train_tensors.append(self.print_lr_op)
            if self.print_gradients:
                train_tensors.append(self.print_gradients_op)
            if self.print_grad_norm:
                train_tensors.append(self.print_grad_norm_op)
            if self.print_labels:
                train_tensors.append(self.print_labels_op)
        try:
            with tqdm(range(num_train_batches), unit="batch") as timer:
                while True:
                    _, batch_no = self.sess.run([self.increment_batch_no_op, self.batch_no])
                    mean_loss, output, mean_cer, *_ = self.sess.run(train_tensors,
                                                                    feed_dict={"ph_batch_size:0": self.batch_size,
                                                                               "ph_is_train:0": True,
                                                                               "ph_ff_dropout:0": self.ff_dropout})
                    timer.update(1)
                    if batch_no % 10 == 0:
                        print("BATCH {} | Loss {} | Error {}".format(batch_no, mean_loss, mean_cer))
        except tf.errors.OutOfRangeError:
            # update train summary
            summary, total_loss, epoch_mean_cer = self.sess.run([self.merged_summaries,
                                                                 self.total_loss,
                                                                 self.epoch_mean_cer])
            self.sw_train.add_summary(summary, epoch)
            # print results to console
            print("Total Loss: {}".format(total_loss))
            print("Mean CER: {}".format(epoch_mean_cer))
            print("Output Example: {}".format("".join([DataLoader.n2c_map[c] for c in output[0][0, :] if c != -1])))
            # reset summary variables
            self.sess.run([self.reset_total_loss, self.reset_batch_no])

        # TESTING Dataset
        self.sess.run(self.inputs["init_test"])
        print("\n_____TESTING DATA_____")
        test_tensors = [self.outputs["mean_loss"],
                        self.outputs["ctc_outputs"],
                        self.outputs["mean_cer"],
                        self.update_epoch_mean_cer,
                        self.increment_total_loss]
        if self.config["debug"]:
            if self.print_labels:
                train_tensors.append(self.print_labels_op)
            if self.print_dropout:
                train_tensors.append(self.print_dropout_op)
        try:
            with tqdm(range(num_test_batches), unit="batch") as timer:
                while True:
                    _, batch_no = self.sess.run([self.increment_batch_no_op, self.batch_no])
                    mean_loss, output, mean_cer, *_ = self.sess.run(test_tensors,
                                                                    feed_dict={"ph_batch_size:0": self.batch_size,
                                                                               "ph_is_train:0": False,
                                                                               "ph_ff_dropout:0": [0.0, 0.0, 0.0]})
                    timer.update(1)
                    if batch_no % 5 == 0:
                        print("BATCH {} | Loss {} | Error {}".format(batch_no, mean_loss, mean_cer))
        except tf.errors.OutOfRangeError:
            # update test summary
            summary, total_loss, epoch_mean_cer = self.sess.run([self.merged_summaries,
                                                                 self.total_loss,
                                                                 self.epoch_mean_cer])
            self.sw_test.add_summary(summary, epoch)
            # print results to console
            print("Total Loss: {}".format(total_loss))
            print("Mean CER: {}".format(epoch_mean_cer))
            print("Output Example: {}".format("".join([DataLoader.n2c_map[c] for c in output[0][0, :] if c != -1])))
            # reset summary variables
            self.sess.run([self.reset_total_loss, self.reset_batch_no])

        # increment global step by one
        self.sess.run(self.increment_global_step_op)

    def train(self, save_every=1):
        # This function is usually common to all your models, Here is an example:
        self.sess.run(self.global_step.initializer)
        for epoch_id in range(0, self.max_epochs):
            self.learn_from_epoch()

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and epoch_id % save_every == 0:
                self.save()

    def infer(self, audio):
        print("\n_____LOADING AUDIO_____")
        signal, fs = load_speech(audio)

        print("\n_____CONVERTING TO MFCC_____")
        m = MFCC([signal], fs)
        cepstrum = m.transform_data(deltas=self.mfcc_deltas)

        size_x = np.array([cepstrum[0].shape[0]], dtype=np.int32)

        # pad to max frame length and reshape into numpy array [1, max_time, num_features]
        try:
            x = list_to_padded_array(cepstrum, size_x[0])  # TODO: save max time during training and load during inference
        except ValueError:
            print("AudioLengthException: The audio to infer is longer than max_time at training.", file=sys.stderr)
            return

        print("\n_____STARTING INFERENCE_____")
        output = self.sess.run(self.outputs["ctc_outputs"],
                               feed_dict={"ph_x:0": x,
                                          "ph_y:0": np.zeros((1, 1), dtype=np.int32),
                                          "ph_size_x:0": size_x,
                                          "ph_batch_size:0": 1,
                                          "ph_is_train:0": False,
                                          "ph_ff_dropout:0": [0.0, 0.0, 0.0]})

        print("\n_____TRANSCRIPT_____:\n{}".format("".join([DataLoader.n2c_map[c] for c in output[0][0, :] if c != -1])))

    def save(self):
        # This function is usually common to all your models, Here is an example:
        global_step_t = tf.train.get_global_step(self.graph)
        global_step = self.sess.run(global_step_t)
        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.checkpoint_save_path, global_step))
        self.saver.save(self.sess, self.checkpoint_save_path + '/epoch-', global_step)

        # I always keep the configuration that
        if not os.path.isfile(self.checkpoint_save_path + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.checkpoint_save_path + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if checkpoint is None or not self.from_checkpoint:
            self.sess.run(self.init_op)
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.checkpoint_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
