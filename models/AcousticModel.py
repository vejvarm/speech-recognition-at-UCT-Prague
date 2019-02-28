# A big thanks for basic layout of this class to:
# Morgan
# TensorFlow: A proposal of good practices for files, folders and models architecture
# https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
import copy
import json
import os
import sys
from datetime import datetime
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from DataLoader import DataLoader
from MFCC import MFCC
from helpers import check_equal, load_config, get_available_devices

load_cepstra = MFCC.load_cepstra  # for loading  cepstrum-###.npy files from a folder into a list of lists
load_labels = DataLoader.load_labels  # for loading transcript-###.npy files from folder into a list of lists


# TODO: try using tf.contrib.rnn.GridLSTMCell instead of stacked bidirectional LSTM
class AcousticModel(object):
    load_dir: str
    save_dir: str
    do_train: bool
    from_checkpoint: bool
    num_cpu_cores: int
    parallel_iterations: int
    cepstrum_pad_val: float
    label_pad_val: int
    num_data: int
    max_time: int
    num_features: int
    ds_train: tf.data.Dataset
    ds_test: tf.data.Dataset
    inputs: Dict[tf.Tensor, tf.Operation]
    outputs: Dict[tf.Tensor, tf.Operation]
    is_train: bool
    global_step: tf.Tensor
    increment_global_step_op: tf.Operation
    lr: float
    max_epochs: int
    batch_size: int
    tt_ratio: float
    shuffle_seed: int
    ff_num_hidden: List[int]
    ff_dropout: List[float]
    ff_batch_norm: bool
    relu_clip: float
    num_hidden: List[int]
    use_peephole: bool
    beam_width: int
    top_paths: int
    grad_clip: bool
    grad_clip_val: float
    show_device_placement: bool
    print_batch_x: bool
    print_layer_3: bool
    print_grad_norm: bool
    print_labels: bool
    print_batch_x_op: tf.print
    print_layer_3_op: tf.print
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
        self.num_cpu_cores = self.config['num_cpu_cores']  # number of CPU cores to use for parallelization
        self.parallel_iterations = self.config['parallel_iterations']  # GPU parallelization in stacked_dynamic_BiRNN
        self.cepstrum_pad_val = self.config['cepstrum_pad_val']  # value with which to pad cepstra to same length
        self.label_pad_val = self.config['label_pad_val']  # value to pad batches of labels to same length
        self.init_op = None

        # Data-inferred parameters (check load_data())#
        self.num_data = None  # total number of individual data in the loaded dataset
        self.max_time = None  # maximal time unrolling of the BiRNN
        self.num_features = None  # number of features in the loaded MFCC cepstra
        self.ds_train = None  # tf.Dataset object with elements of training data with components (cepstrum, label)
        self.ds_test = None  # tf.Dataset object with elements of testing data with components (cepstrum, label)
        self.inputs = None  # dictionary with the inputs to model from batched dataset iterators
        self.outputs = None  # dictionary with outputs from the model
        self.is_train = None  # (bool) placeholder in which we feed True during training and False otherwise
        self.global_step = None  # (int) counter for current epoch of training
        self.increment_global_step_op = None  # operation for incrementing global_step by 1

        # HyperParameters (HP) #
        # size of the alphabet in DataLoader
        self.alphabet_size = len(DataLoader.c2n_map)  # number of characters in the alphabet of transcripts
        if config['random']:  # TODO: The missing (None) params will be configured randomly
            pass  # TODO: random_config() for generating random configurations of the hyperparams
        else:
            # training HP
            self.lr = self.config['lr']  # (float) learning rate
            self.max_epochs = self.config['max_epochs']  # (int) maximum number of training epochs
            self.batch_size = self.config['batch_size']  # (int) size of mini_batches to be fed into the net at once
            self.tt_ratio = self.config['tt_ratio']  # (float) train-test data split ratio
            self.shuffle_seed = self.config['shuffle_seed']  # (int) seed for shuffling the cepstra and labels

            # AcousticModel specific HP
            self.ff_num_hidden = self.config['ff_num_hidden']  # (list of ints) hidden units in feed forward layers
            self.ff_dropout = self.config['ff_dropout']  # (list of floats) dropouts after each feed forward layer
            self.ff_batch_norm = self.config['ff_batch_norm']  # (bool) use batch normalisation in feed forward layers
            self.relu_clip = self.config['relu_clip']  # (float) preventing exploding gradient with relu clipping
            self.num_hidden = self.config['num_hidden']  # (list of ints) number of hidden units in LSTM cells
            self.use_peephole = self.config['use_peephole']  # (bool) use peephole connections in the LSTM cells
            self.beam_width = self.config['beam_width']  # (int) beam width for the Beam Search (BS) algorithm
            self.top_paths = self.config['top_paths']  # (int) number of best paths to return from the BS algorithm
            self.grad_clip = self.config['grad_clip']  # (bool) if or not to use gradient clipping by global norm
            self.grad_clip_val = self.config['grad_clip_val']  # (float) value at which to clip gradient global norm

        # DEBUGGING SETTINGS # (works only if config["debug"] == True)
        self.show_device_placement = self.config['show_device_placement']
        self.print_batch_x = self.config['print_batch_x']
        self.print_layer_3 = self.config['print_layer_3']
        self.print_grad_norm = self.config['print_grad_norm']  # (bool) whether to print out gradient norm at each batch
        self.print_labels = self.config['print_labels']

        # PRINT OPERATIONS #
        self.print_batch_x_op = None
        self.print_layer_3_op = None
        self.print_grad_norm_op = None
        self.print_labels_op = None

        self.episode_id = datetime.now().strftime('%Y%m-%d%H-%M%S')  # unique episode id from current date and time

        # TODO: Set properties/hyperparams function (overriding the 'config' dictionary)

        # TODO: save_configuration()

        self.graph = tf.Graph()

        # load_data()
        self.load_data()
        # prepare data
        self.prepare_data()

        # TODO: build_graph()
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
        self.sw = tf.summary.FileWriter(self.save_dir, self.sess.graph)

        # TODO: learn_from_epoch()

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
        # get maximum number of frames which will serve as a max_time for unrolling the dynamic_rnn
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

        with self.graph.as_default():
            # combine the elements in datasets into batches of padded components
            padded_shapes = (tf.TensorShape([self.max_time, self.num_features]),  # cepstra padded to self.max_time
                             tf.TensorShape([None]),  # labels padded to max length in batch
                             tf.TensorShape([]),  # sizes not padded
                             tf.TensorShape([]))  # sizes not padded
            padding_values = (tf.constant(self.cepstrum_pad_val, dtype=tf.float32),  # cepstra padded with 0.0
                              tf.constant(self.label_pad_val, dtype=tf.int32),  # labels padded with -1
                              0,  # size(cepstrum) -- unused
                              0)  # size(label) -- unused

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
    def ff_layer(w_name, b_name, input_size, output_size):
        w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.0001), name=w_name)
        b = tf.Variable(tf.truncated_normal([output_size], stddev=0.0001), name=b_name)
        return w, b

    @staticmethod
    def batch_norm_layer(x, is_train, scope):
        # !!! during training the tf.GraphKeys.UPDATE_OPS must be called to update the mean and variance
        bn_train = tf.contrib.layers.batch_norm(x,
                                                decay=0.999,
                                                is_training=True,
                                                scope=scope,
                                                updates_collections=None)
        bn_infer = tf.contrib.layers.batch_norm(x,
                                                decay=0.999,
                                                is_training=False,
                                                reuse=True,
                                                scope=scope,
                                                updates_collections=None)
        return tf.cond(is_train, lambda: bn_train, lambda: bn_infer)

    def lstm_cell(self, num_hidden):
        # TODO: try to use tf.contrib.cudnn_rnn.CudnnLSTM(num_units=self.num_hidden, state_is_tuple=True)
        # TODO: Batch Normalization at LSTM outputs (before the activation function)
        return tf.contrib.rnn.LSTMBlockCell(num_units=num_hidden, use_peephole=self.use_peephole)
        # return tf.contrib.grid_rnn.Grid1LSTMCell(num_units=num_hidden, state_is_tuple=True)
        # return tf.contrib.rnn.GridLSTMCell(num_units=num_hidden, state_is_tuple=True, num_frequency_blocks=None)
        # return tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True, activation='tanh')

    def build_graph(self):
        # TODO: inputs and labels
        # x_placeholder = tf.placeholder(tf.float32, (None, None, self.num_features))
        # y_placeholder = tf.placeholder(tf.int32, (None, None))

        devices = get_available_devices()

        with self.graph.as_default():

            self.is_train = is_train = tf.placeholder(tf.bool, name="is_train")

            # reshaping from [batch_size, max_time, num_features] to [max_time*batch_size, num_features]
            with tf.device(devices["cpu"][0]):
                batch_x = tf.transpose(self.inputs["x"], [1, 0, 2])  # transpose to [max_time, batch_size, num_features]
                batch_x = tf.reshape(batch_x, [-1, self.num_features])  # reshape to [max_time*batch_size, num_features]

                if self.print_batch_x:
                    # print batch_x
                    self.print_batch_x_op = tf.print("batch_x: ", batch_x, "shape: ", batch_x.shape,
                                                     output_stream=sys.stdout)

                # feed forward layers
                # 1st layer
                with tf.variable_scope("layer_1") as scope:
                    w1, b1 = self.ff_layer("w1", "b1", self.num_features, self.ff_num_hidden[0])
                    layer_1 = tf.add(tf.matmul(batch_x, w1), b1)
                    if self.ff_batch_norm:
                        layer_1 = self.batch_norm_layer(layer_1, is_train, scope)
                    layer_1 = tf.minimum(tf.nn.relu(layer_1), self.relu_clip)
                    layer_1 = tf.nn.dropout(layer_1, keep_prob=(1.0 - self.ff_dropout[0]))

                # 2nd layer
                with tf.variable_scope("layer_2") as scope:
                    w2, b2 = self.ff_layer("w2", "b2", self.ff_num_hidden[0], self.ff_num_hidden[1])
                    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
                    if self.ff_batch_norm:
                        layer_2 = self.batch_norm_layer(layer_2, is_train, scope)
                    layer_2 = tf.minimum(tf.nn.relu(layer_2), self.relu_clip)
                    layer_2 = tf.nn.dropout(layer_2, keep_prob=(1.0 - self.ff_dropout[1]))

                # 3rd layer
                with tf.variable_scope("layer_3") as scope:
                    w3, b3 = self.ff_layer("w3", "b3", self.ff_num_hidden[1], self.ff_num_hidden[2])
                    layer_3 = tf.add(tf.matmul(layer_2, w3), b3)
                    if self.ff_batch_norm:
                        layer_3 = self.batch_norm_layer(layer_3, is_train, scope)
                    layer_3 = tf.minimum(tf.nn.relu(layer_3), self.relu_clip)
                    layer_3 = tf.nn.dropout(layer_3, keep_prob=(1.0 - self.ff_dropout[2]))
                    # reshape back to [batch_size, max_time, ff_num_hidden[2]] for the dynamic rnn
                    layer_3 = tf.reshape(layer_3, [self.batch_size, self.max_time, self.ff_num_hidden[2]])

                if self.print_layer_3:
                    self.print_layer_3_op = tf.print("layer_3: ", layer_3, "shape: ", layer_3.shape,
                                                     output_stream=sys.stdout)

            # 4th layer: stacked BiRNN with LSTM cells
            # TODO: LSTM might become a computational bottleneck
            cells_fw = [self.lstm_cell(n) for n in self.num_hidden]  # list of forward direction cells
            cells_bw = [self.lstm_cell(n) for n in self.num_hidden]  # list of backward direction cells
            rnn_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                               cells_bw,
                                                                               inputs=layer_3,  # TODO: Check the layers
                                                                               sequence_length=self.inputs["size_x"],
                                                                               dtype=tf.float32,
                                                                               parallel_iterations=self.parallel_iterations)
            # rnn_outputs == Tensor of shape [batch_size, max_time, 2*num_hidden]

            # transpose rnn_outputs into time major tensor -> [max_time, batch_size, 2*num_hidden]
            rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])

            # Reshape output from a tensor of shape [max_time, batch_size, 2*num_hidden]
            # to a tensor of shape [max_time*batch_size, 2*num_hidden]
            rnn_outputs = tf.reshape(rnn_outputs, [-1, 2 * self.num_hidden[-1]])

            # 2nd layer: linear projection of outputs from BiRNN
            # define weights and biases for linear projection of outputs from BiRNN
            logit_size = self.alphabet_size + 1  # +1 for the blank
            lp_weights = tf.Variable(tf.random.normal([2 * self.num_hidden[-1], logit_size], dtype=tf.float32))
            lp_biases = tf.Variable(tf.random.normal([logit_size], dtype=tf.float32))

            # convert rnn_outputs into logits (apply linear projection of rnn outputs)
            # lp_outputs.shape == [max_time*batch_size, alphabet_size + 1]
            lp_outputs = tf.minimum(tf.nn.relu(tf.add(tf.matmul(rnn_outputs, lp_weights), lp_biases)), self.relu_clip)

            # reshape lp_outputs to shape [max_time, batch_size, alphabet_size + 1]
            logits = tf.reshape(lp_outputs, [self.max_time, self.batch_size, logit_size])

            # switch the batch_size and max_time dimensions (ctc inputs must be time major)
            #        logits = tf.transpose(logits, perm=[1, 0, 2])

            # print labels
            if self.print_labels:
                self.print_labels_op = tf.print("labels: ", self.inputs["y"], output_stream=sys.stdout)

            # convert labels to sparse tensor
            labels = tf.contrib.layers.dense_to_sparse(self.inputs["y"], eos_token=self.label_pad_val)

            # decode the logits
            # TODO: create separate function for this
            ctc_outputs, ctc_log_probs = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                                       sequence_length=self.inputs["size_x"],
                                                                       beam_width=self.beam_width,
                                                                       top_paths=self.top_paths,
                                                                       merge_repeated=True)

            # convert outputs from sparse to dense
            ctc_outputs = [tf.sparse.to_dense(output, default_value=self.label_pad_val) for output in ctc_outputs]

            # calculate ctc loss of logits
            ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=logits,
                                      sequence_length=self.inputs["size_x"])

            # Calculate the average loss across the batch
            avg_loss = tf.reduce_mean(ctc_loss)

            # use AdamOptimizer to comput the gradients and minimize the average of ctc_loss (training the model)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            gradients, variables = zip(*optimizer.compute_gradients(avg_loss))
            if self.grad_clip:
                gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_val)
            if self.print_grad_norm:
                self.print_grad_norm_op = tf.print("Grad. global norm:", tf.global_norm(gradients),
                                                   output_stream=sys.stdout)
            optimize = optimizer.apply_gradients(zip(gradients, variables))

            self.outputs = {"ctc_outputs": ctc_outputs,
                            "ctc_log_probs": ctc_log_probs,
                            "ctc_loss": ctc_loss,
                            "avg_loss": avg_loss,
                            "optimize": optimize}

            # global step tensor
            self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

            # operation for incrementing global step
            self.increment_global_step_op = tf.assign_add(self.global_step, 1)

            # initializer for TensorFlow variables
            self.init_op = tf.global_variables_initializer()

            # tf.nn.batch_normalization()
            # TODO: optimizer
            # TODO: minimize(cost)
            # raise Exception('The build_graph function must be implemented')

    def infer(self):
        raise Exception('The infer function must be implemented')

    def learn_from_epoch(self):
        total_train_loss = 0
        count_train = 0
        total_test_loss = 0
        count_test = 0
        output = None

        num_train_batches = int(self.num_data * self.tt_ratio // self.batch_size)
        num_test_batches = int(self.num_data * (1 - self.tt_ratio) // self.batch_size)

        # increment global step by one
        self.sess.run(self.increment_global_step_op)

        # TRAINING Dataset
        self.sess.run(self.inputs["init_train"])
        print("_____TRAINING DATA_____")
        train_tensors = [self.outputs["optimize"],
                         self.outputs["ctc_loss"],
                         self.outputs["avg_loss"],
                         self.outputs["ctc_outputs"]]
        if self.config["debug"]:
            if self.print_batch_x:
                train_tensors.append(self.print_batch_x_op)
            if self.print_layer_3:
                train_tensors.append(self.print_layer_3_op)
            if self.print_grad_norm:
                train_tensors.append(self.print_grad_norm_op)
            if self.print_labels:
                train_tensors.append(self.print_labels_op)
        try:
            with tqdm(range(num_train_batches), unit="batch") as timer:
                while True:
                    _, ctc_loss, avg_loss, output, *_ = self.sess.run(train_tensors, feed_dict={self.is_train: True})
                    total_train_loss += avg_loss
                    count_train += 1
                    timer.update(1)
                    if count_train % 10 == 0:
                        print("BATCH {} | Avg. Loss {}".format(count_train, avg_loss))
        except tf.errors.OutOfRangeError:
            print("Total Loss: {}".format(total_train_loss))
            print("Output Example: {}".format("".join([DataLoader.n2c_map[c] for c in output[0][0, :] if c != -1])))

        # TESTING Dataset
        self.sess.run(self.inputs["init_test"])
        print("_____TESTING DATA_____")
        try:
            with tqdm(range(num_test_batches), unit="batch") as timer:
                while True:
                    ctc_loss, avg_loss, output = self.sess.run([self.outputs["ctc_loss"],
                                                                self.outputs["avg_loss"],
                                                                self.outputs["ctc_outputs"]],
                                                               feed_dict={self.is_train: False})
                    total_test_loss += avg_loss
                    count_test += 1
                    timer.update(1)
                    if count_test % 5 == 0:
                        print("BATCH {} | Avg. Loss {}".format(count_test, avg_loss))
        except tf.errors.OutOfRangeError:
            print("Total Loss: {}".format(total_test_loss))
            print("Output Example: {}".format("".join([DataLoader.n2c_map[c] for c in output[0][0, :] if c != -1])))

    def train(self, save_every=1):
        # This function is usually common to all your models, Here is an example:
        self.sess.run(self.global_step.initializer)
        self.episode_id = datetime.now().strftime('%Y%m-%d%H-%M%S')  # unique episode id from current date and time
        for epoch_id in range(0, self.max_epochs):
            self.learn_from_epoch()

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and epoch_id % save_every == 0:
                self.save()

    def save(self):
        # This function is usually common to all your models, Here is an example:
        global_step_t = tf.train.get_global_step(self.graph)
        global_step = self.sess.run(global_step_t)
        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.save_dir, global_step))
        self.saver.save(self.sess, self.save_dir + '/agent-ep_' + str(self.episode_id), global_step)

        # I always keep the configuration that
        if not os.path.isfile(self.save_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.save_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint is None or not self.from_checkpoint:
            self.sess.run(self.init_op)
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.save_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
