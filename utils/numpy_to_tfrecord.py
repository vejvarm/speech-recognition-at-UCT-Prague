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


def numpy_to_tfrecord(path_to_files, output_path='',
                      feature_names='cepstrum', label_names='transcript'):
    """

    :param path_to_files: (str) path to folder with subfolders of numpy files
    :param output_path: (str)path to folder with output tfrecord files
    :param feature_names: sequence of symbols that can be used as common identifier for feature files
    :param label_names: sequence of symbols that can be used as common identifier for label files
    :return: None
    """

    path_to_files = os.path.normpath(path_to_files)
    folder_structure_gen = os.walk(path_to_files)  # ('path_to_current_folder', [subfolders], ['files', ...])

    if output_path and isinstance(output_path, str):
        output_path = os.path.normpath(output_path)
    else:
        output_path = path_to_files

    for folder in folder_structure_gen:
        path, subfolders, files = folder
        if not files:
            continue
        feat_file_names = [f for f in files if feature_names in f]
        label_file_names = [f for f in files if label_names in f]

        num_feats = len(feat_file_names)
        num_labels = len(label_file_names)

        assert num_feats == num_labels, 'There is {} feature files and {} label files (must be same).'.format(num_feats,
                                                                                                              num_labels)

        tfrecord_path = os.path.join(output_path, os.path.split(path)[1]) + '.tfrecord'
        writer = tf.python_io.TFRecordWriter(tfrecord_path)

        for i in range(num_feats):
            feat_load_path = os.path.join(path, feat_file_names[i])
            label_load_path = os.path.join(path, label_file_names[i])

            feat, _ = FeatureExtractor.load_cepstra(feat_load_path)
            label, _ = DataLoader.load_labels(label_load_path)

            #            print(feat[0][0].shape, label[0][0].shape)

            serialized = serialize_array(feat[0][0], label[0][0])
            writer.write(serialized)

        writer.close()

        print("Data written to {}".format(tfrecord_path))


if __name__ == '__main__':
    path_to_files = ["b:/!temp/ORAL_MFSC_unigram_40_banks_min_100_max_3000/train",
                     "b:/!temp/ORAL_MFSC_unigram_40_banks_min_100_max_3000/test"]

    for path in path_to_files:
        numpy_to_tfrecord(path)
