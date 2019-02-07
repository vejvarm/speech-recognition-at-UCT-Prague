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
        # check if inputs :ivars are dictionaries
        self.assertIsInstance(self.ac_model.inputs_train, dict)
        self.assertIsInstance(self.ac_model.inputs_test, dict)

        # check if all the keys are present in inputs_train dict
        self.assertIn("x", self.ac_model.inputs_train)
        self.assertIn("y", self.ac_model.inputs_train)
        self.assertIn("size_x", self.ac_model.inputs_train)
        self.assertIn("size_y", self.ac_model.inputs_train)
        self.assertIn("iterator_init", self.ac_model.inputs_train)

        # check if all the keys are present in inputs_train dict
        self.assertIn("x", self.ac_model.inputs_test)
        self.assertIn("y", self.ac_model.inputs_test)
        self.assertIn("size_x", self.ac_model.inputs_test)
        self.assertIn("size_y", self.ac_model.inputs_test)
        self.assertIn("iterator_init", self.ac_model.inputs_test)

        outputs = (self.ac_model.inputs_train["x"],
                   self.ac_model.inputs_train["y"],
                   self.ac_model.inputs_train["size_x"],
                   self.ac_model.inputs_train["size_y"])

        with tf.Session() as sess:
            sess.run(self.ac_model.inputs_train["iterator_init"])
            x, y, size_x, size_y = sess.run(outputs)
            print(x, y, size_x, size_y)