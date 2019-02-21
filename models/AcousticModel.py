# A big thanks for basic layout of this class to:
# Morgan
# TensorFlow: A proposal of good practices for files, folders and models architecture
# https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
import os
import copy
import json

from datetime import datetime

import numpy as np
import tensorflow as tf

from helpers import check_equal, random_shuffle, load_config
from MFCC import MFCC
from DataLoader import DataLoader
load_cepstra = MFCC.load_cepstra  # for loading  cepstrum-###.npy files from a folder into a list of lists
load_labels = DataLoader.load_labels  # for loading transcript-###.npy files from folder into a list of lists


# TODO: try using tf.contrib.rnn.GridLSTMCell instead of stacked bidirectional LSTM
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
        self.from_checkpoint = self.config['from_checkpoint']  # if True, loads last checkpoint in save_dir
        self.num_cpu_cores = self.config['num_cpu_cores']  # number of CPU cores to use for parallelization
        self.parallel_iterations = self.config['parallel_iterations']  # GPU parallelization in stacked_dynamic_BiRNN
        self.cepstrum_pad_val = self.config['cepstrum_pad_val']  # value with which to pad cepstra to same length
        self.label_pad_val = self.config['label_pad_val']        # value to pad batches of labels to same length
        self.init_op = None

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
            self.use_peephole = self.config['use_peephole']  # whether to use peephole connections in the LSTM cells
            self.beam_width = self.config['beam_width']   # beam width for the Beam Search (BS) algorithm
            self.top_paths = self.config['top_paths']     # number of best paths to return from the BS algorithm

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
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
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
        len_train = int(self.tt_ratio*self.num_data)  # length of the training data
        len_test = self.num_data - int(self.tt_ratio*self.num_data)  # length of the testing data
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
                             tf.TensorShape([None]),                              # labels padded to max length in batch
                             tf.TensorShape([]),                                  # sizes not padded
                             tf.TensorShape([]))                                  # sizes not padded
            padding_values = (tf.constant(self.cepstrum_pad_val, dtype=tf.float32),  # cepstra padded with 0.0
                              tf.constant(self.label_pad_val, dtype=tf.int32),       # labels padded with -1
                              0,                                                  # size(cepstrum) -- unused
                              0)                                                  # size(label) -- unused

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

        with self.graph.as_default():
            # 1st layer: stacked BiRNN with LSTM cells
            # TODO: consider using tf.contrib.rnn.stack_bidirectional_dynamic_rnn
            cells_fw = [self.lstm_cell(n) for n in self.num_hidden]  # list of forward direction cells
            cells_bw = [self.lstm_cell(n) for n in self.num_hidden]  # list of backward direction cells
            rnn_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                               cells_bw,
                                                                               inputs=self.inputs["x"],
                                                                               sequence_length=self.inputs["size_x"],
                                                                               dtype=tf.float32,
                                                                               parallel_iterations=self.parallel_iterations)
            # rnn_outputs == Tensor of shape [batch_size, max_time, 2*num_hidden]

            # transpose rnn_outputs into time major tensor -> [max_time, batch_size, 2*num_hidden]
            rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])

            # Reshape output from a tensor of shape [max_time, batch_size, 2*num_hidden]
            # to a tensor of shape [max_time*batch_size, 2*num_hidden]
            rnn_outputs = tf.reshape(rnn_outputs, [-1, 2*self.num_hidden[-1]])

            # 2nd layer: linear projection of outputs from BiRNN
            # define weights and biases for linear projection of outputs from BiRNN
            logit_size = self.alphabet_size + 1  # +1 for the blank
            lp_weights = tf.Variable(tf.random.normal([2*self.num_hidden[-1], logit_size], dtype=tf.float32))
            lp_biases = tf.Variable(tf.random.normal([logit_size], dtype=tf.float32))

            # convert rnn_outputs into logits (apply linear projection of rnn outputs)
            # lp_outputs.shape == [max_time*batch_size, alphabet_size + 1]
            lp_outputs = tf.nn.relu(tf.add(tf.matmul(rnn_outputs, lp_weights), lp_biases))

            # reshape lp_outputs to shape [max_time, batch_size, alphabet_size + 1]
            logits = tf.reshape(lp_outputs, [self.max_time, self.batch_size, logit_size])

            # switch the batch_size and max_time dimensions (ctc inputs must be time major)
    #        logits = tf.transpose(logits, perm=[1, 0, 2])

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

            # use AdamOptimizer to minimize the ctc_losses (training the model)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(avg_loss)

            self.outputs = {"ctc_outputs": ctc_outputs,
                            "ctc_log_probs": ctc_log_probs,
                            "ctc_loss": ctc_loss,
                            "avg_loss": avg_loss,
                            "optimizer": optimizer}

            # global step tensor
            self.global_step = tf.Variable(1, trainable=False, name='global_step', dtype=tf.int32)

            # operation for incrementing global step
            self.increment_global_step_op = tf.assign_add(self.global_step, 1)

            # initializer for TensorFlow variables
            self.init_op = tf.global_variables_initializer()

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
        total_train_loss = 0
        count_train = 0
        total_test_loss = 0
        count_test = 0

        # TRAINING Dataset
        self.sess.run(self.inputs["init_train"])
        print("_____TRAINING DATA_____")
        try:
            while True:
                _, ctc_loss, avg_loss, output = self.sess.run([self.outputs["optimizer"],
                                                               self.outputs["ctc_loss"],
                                                               self.outputs["avg_loss"],
                                                               self.outputs["ctc_outputs"]])
                total_train_loss += avg_loss
                count_train += 1
                if count_train % 2 == 0:
                    print("BATCH {} | Avg. Loss {}".format(count_train, avg_loss))
        except tf.errors.OutOfRangeError:
            print("Total Loss: {}".format(total_train_loss))
            print("Output Example: {}".format(output))

        # TESTING Dataset
        self.sess.run(self.inputs["init_test"])
        print("_____TESTING DATA_____")
        try:
            while True:
                ctc_loss, avg_loss, output = self.sess.run([self.outputs["ctc_loss"],
                                                            self.outputs["avg_loss"],
                                                            self.outputs["ctc_outputs"]])
                total_test_loss += avg_loss
                count_test += 1
                print("BATCH {} | Avg. Loss {}".format(count_test, avg_loss))
        except tf.errors.OutOfRangeError:
            print("Total Loss: {}".format(total_test_loss))
            print("Output Example: {}".format(output))

        # increment global step by one
        self.sess.run(self.increment_global_step_op)
        # raise Exception('The learn_from_epoch function must be implemented')

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



