import os

import tensorflow as tf
import numpy as np

from FeatureExtraction import FeatureExtractor
from DataLoader import DataLoader


def serialize_array(x, y):
    feature = {
        'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten())),
        'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def numpy_to_tfrecord(path_to_files, output_file=''):
    """

    :param path_to_files: (str) path to folder with numpy files
    :param output_file: (str)path to output tfrecord file
    :return: None
    """

    path_to_files = os.path.normpath(path_to_files)

    if output_file and isinstance(output_file, str):
        output_file = os.path.normpath(output_file)
    else:
        output_file = os.path.join(path_to_files, 'data.tfrecord')

    cepstra, cepstra_paths = FeatureExtractor.load_cepstra(path_to_files)
    labels, label_paths = DataLoader.load_labels(path_to_files)

    num_cepstra = len(cepstra[0])
    num_labels = len(labels[0])

    assert num_cepstra == num_labels, 'number of feat files ({}) and label files ({}) must be equal'.format(num_cepstra,
                                                                                                            num_labels)

    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(num_cepstra):
        serialized = serialize_array(cepstra[0][i], labels[0][i])
        writer.write(serialized)

    writer.close()

    "Data written to {}".format(output_file)

if __name__ == '__main__':
    path_to_files = "c:/!temp/MFSC_debug"
    output_file = "c:/!temp/data.tfrecord"

    numpy_to_tfrecord(path_to_files, output_file)
