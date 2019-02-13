from os.path import isdir
from shutil import rmtree
from unittest import TestCase

from models.AcousticModel import AcousticModel

import tensorflow as tf


class TestAcousticModel(TestCase):
    def setUp(self):
        self.ac_model = AcousticModel("../config")

class TestData(TestAcousticModel):

    def test_prepare_data(self):

        count_train = 0
        count_test = 0

        # check if inputs :ivars are dictionaries
        self.assertIsInstance(self.ac_model.inputs, dict)

        # check if all the keys are present in inputs dict
        self.assertIn("x", self.ac_model.inputs)
        self.assertIn("y", self.ac_model.inputs)
        self.assertIn("size_x", self.ac_model.inputs)
        self.assertIn("size_y", self.ac_model.inputs)
        self.assertIn("init_train", self.ac_model.inputs)
        self.assertIn("init_test", self.ac_model.inputs)

        outputs = (self.ac_model.inputs["x"],
                   self.ac_model.inputs["y"],
                   self.ac_model.inputs["size_x"],
                   self.ac_model.inputs["size_y"])

        with tf.Session() as sess:
            # training dataset
            sess.run(self.ac_model.inputs["init_train"])
            try:
                while True:
                    x, y, size_x, size_y = sess.run(outputs)
                    count_train += 1
                    if count_train % 10 == 0:
                        print("___TRAINING BATCH no. {}___\n".format(count_train), x, y, size_x, size_y)
            except tf.errors.OutOfRangeError:
                pass

            # testing dataset
            sess.run(self.ac_model.inputs["init_test"])
            try:
                while True:
                    x, y, size_x, size_y = sess.run(outputs)
                    count_test += 1
                    if count_test % 10 == 0:
                        print("___TESTING BATCH no. {}___\n".format(count_test), x, y, size_x, size_y)

            except tf.errors.OutOfRangeError:
                pass

        print("Number of train batches: %u", count_train)
        print("Number of test batches: %u", count_test)

    def test_build_graph(self):

        total_loss = 0
        count_train = 0
        count_test = 0

        with tf.Session() as sess:
            # training dataset
            sess.run(tf.global_variables_initializer())
            sess.run(self.ac_model.inputs["init_train"])
            try:
                while True:
                    output = sess.run(self.ac_model.outputs["ctc_output"])
                    total_loss += 0
                    count_train += 1
                    if count_train % 10 == 0:
                        print("___TRAINING BATCH no. {}___\n".format(count_train), output, 0)
            except tf.errors.OutOfRangeError:
                pass


